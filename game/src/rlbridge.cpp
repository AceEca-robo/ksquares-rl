/*
    RL bridge: see rlbridge.h

    SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "rlbridge.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSocketNotifier>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "ksquaresgame.h"
#include "ksquareswindow.h"
#include "settings.h"

int RLBridge::aiDelayMs()
{
    bool ok = false;
    const int ms = qEnvironmentVariableIntValue("KSQUARES_RL_AI_DELAY_MS", &ok);
    return ok && ms > 0 ? ms : 0;
}

RLBridge *RLBridge::start(KSquaresWindow *window, KSquaresGame *game)
{
    const QByteArray path = qgetenv("KSQUARES_RL_SOCKET");
    if (path.isEmpty()) {
        return nullptr;
    }
    struct sockaddr_un addr;
    if (size_t(path.size()) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "rl: socket path too long\n");
        exit(1);
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        perror("rl: socket");
        exit(1);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, path.constData(), path.size());
    if (::connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("rl: connect");
        exit(1);
    }
    return new RLBridge(window, game, fd);
}

RLBridge::RLBridge(KSquaresWindow *window, KSquaresGame *game, int fd)
    : QObject(window), m_window(window), m_game(game), m_fd(fd)
{
    // every drawn line, with the player who drew it (the turn passes later,
    // in KSquaresGame::tryEndGo)
    connect(m_game, &KSquaresGame::drawLine, this, [this](int index) {
        m_moves.append(QJsonArray{index, m_game->currentPlayerId(), QJsonArray()});
    });
    // boxes completed by that line (drawSquare follows its drawLine)
    connect(m_game, &KSquaresGame::drawSquare, this, [this](int index) {
        if (m_moves.isEmpty()) {
            return;
        }
        QJsonArray mv = m_moves.last().toArray();
        QJsonArray boxes = mv.at(2).toArray();
        boxes.append(index);
        mv[2] = boxes;
        m_moves[m_moves.size() - 1] = mv;
    });
    m_notifier = new QSocketNotifier(m_fd, QSocketNotifier::Read, this);
    connect(m_notifier, &QSocketNotifier::activated, this, &RLBridge::onReadable);
    sendJson(QByteArrayLiteral("{\"type\":\"hello\",\"version\":1}"));
}

void RLBridge::sendJson(const QByteArray &json)
{
    QByteArray buf = json + '\n';
    const char *s = buf.constData();
    qsizetype n = buf.size();
    while (n > 0) {
        ssize_t k = ::send(m_fd, s, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) {
            continue;
        }
        if (k <= 0) {
            QCoreApplication::exit(0);
            return;
        }
        s += k;
        n -= k;
    }
}

void RLBridge::sendError(const QString &msg)
{
    QJsonObject o{{QStringLiteral("type"), QStringLiteral("error")}, {QStringLiteral("msg"), msg}};
    sendJson(QJsonDocument(o).toJson(QJsonDocument::Compact));
}

void RLBridge::sendState(bool done)
{
    const QList<bool> lines = m_game->lines();
    QByteArray lineStr(lines.size(), '0');
    for (int i = 0; i < lines.size(); i++) {
        if (lines.at(i)) {
            lineStr[i] = '1';
        }
    }
    QJsonArray owners;
    for (int o : m_game->squares()) {
        owners.append(o);
    }
    QVector<int> scores(Settings::numOfPlayers(), 0);
    for (int o : m_game->squares()) {
        if (o >= 0 && o < scores.size()) {
            scores[o]++;
        }
    }
    QJsonArray scoreArr;
    for (int s : scores) {
        scoreArr.append(s);
    }
    QJsonObject o{
        {QStringLiteral("type"), QStringLiteral("state")},
        {QStringLiteral("moves"), m_moves},
        {QStringLiteral("lines"), QString::fromLatin1(lineStr)},
        {QStringLiteral("owners"), owners},
        {QStringLiteral("scores"), scoreArr},
        {QStringLiteral("player"), m_game->currentPlayerId()},
        {QStringLiteral("done"), done},
    };
    m_moves = QJsonArray();
    m_pending = false;
    sendJson(QJsonDocument(o).toJson(QJsonDocument::Compact));
}

void RLBridge::awaitMove()
{
    m_waitingMove = true;
    if (m_pending) {
        sendState(false);
    }
}

void RLBridge::gameOver()
{
    m_waitingMove = false;
    if (m_pending) {
        sendState(true);
    }
}

void RLBridge::onReadable()
{
    char buf[4096];
    ssize_t k = ::recv(m_fd, buf, sizeof(buf), 0);
    if (k < 0 && (errno == EINTR || errno == EAGAIN)) {
        return;
    }
    if (k <= 0) { // peer gone
        m_notifier->setEnabled(false);
        QCoreApplication::exit(0);
        return;
    }
    m_inbuf.append(buf, int(k));
    int nl;
    while ((nl = m_inbuf.indexOf('\n')) >= 0) {
        QByteArray line = m_inbuf.left(nl);
        m_inbuf.remove(0, nl + 1);
        handle(line);
    }
}

void RLBridge::handle(const QByteArray &line)
{
    QJsonParseError err;
    const QJsonObject o = QJsonDocument::fromJson(line, &err).object();
    if (err.error != QJsonParseError::NoError) {
        sendError(QStringLiteral("bad json: ") + err.errorString());
        return;
    }
    const QString cmd = o.value(QStringLiteral("cmd")).toString();
    if (m_pending) {
        sendError(QStringLiteral("busy"));
        return;
    }
    if (cmd == QLatin1String("quit")) {
        QCoreApplication::exit(0);
    } else if (cmd == QLatin1String("reset")) {
        const int w = o.value(QStringLiteral("w")).toInt(10);
        const int h = o.value(QStringLiteral("h")).toInt(7);
        const QJsonArray ai = o.value(QStringLiteral("ai")).toArray();
        if (w < 1 || h < 1 || w * h > 1500 || ai.size() < 2 || ai.size() > 4) {
            sendError(QStringLiteral("bad reset parameters"));
            return;
        }
        QList<int> human;
        QStringList names;
        for (int i = 0; i < 4; i++) {
            human.append(i < ai.size() && ai.at(i).toBool() ? 0 : 2);
            names.append(QStringLiteral("Player %1").arg(i + 1));
        }
        Settings::setNumOfPlayers(ai.size());
        Settings::setHumanList(human);
        Settings::setPlayerNames(names);
        Settings::setBoardWidth(w);
        Settings::setBoardHeight(h);
        Settings::setQuickStart(0);
        Settings::setDifficulty(o.value(QStringLiteral("difficulty")).toInt(0));
        const QJsonValue seed = o.value(QStringLiteral("seed"));
        if (seed.isDouble()) {
            srand(unsigned(qint64(seed.toDouble())));
        }
        m_moves = QJsonArray();
        m_waitingMove = false;
        m_pending = true;
        m_window->gameReset(); // a human first player answers synchronously
    } else if (cmd == QLatin1String("step")) {
        const int ln = o.value(QStringLiteral("line")).toInt(-1);
        const QList<bool> lines = m_game->lines();
        if (!m_waitingMove) {
            sendError(QStringLiteral("not an external player's turn"));
            return;
        }
        if (ln < 0 || ln >= lines.size() || lines.at(ln)) {
            sendError(QStringLiteral("illegal line %1").arg(ln));
            return;
        }
        m_waitingMove = false;
        m_pending = true;
        m_game->addLineToIndex(ln);
    } else {
        sendError(QStringLiteral("unknown cmd"));
    }
}
