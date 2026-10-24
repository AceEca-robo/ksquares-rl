/*
    RL bridge: drives KSquares from an external agent over a Unix socket.

    SPDX-License-Identifier: GPL-2.0-or-later
*/

#ifndef RLBRIDGE_H
#define RLBRIDGE_H

// RL bridge
// If the environment variable KSQUARES_RL_SOCKET names a Unix socket, the
// game connects to it as a client instead of showing the new game dialog.
// Protocol: one JSON object per line in both directions.
//
//   game -> agent  {"type":"hello","version":1}
//   agent -> game  {"cmd":"reset","w":5,"h":5,"ai":[false,true],
//                   "difficulty":2,"seed":42}
//                  "ai": one flag per player (2..4 players), player 0 moves
//                  first; "seed" (optional) calls srand(seed) before the
//                  game, otherwise the glibc rand() sequence continues.
//   agent -> game  {"cmd":"step","line":17}   (only on an external turn)
//   agent -> game  {"cmd":"quit"}
//   game -> agent  {"type":"state","moves":[[line,player,[boxes]],...],
//                   "lines":"0110...","owners":[-1,0,...],"scores":[..],
//                   "player":p,"done":false}
//                  sent after reset/step once an external player is to move
//                  or the game is over; "moves" are all lines drawn since the
//                  previous state message, with the player who drew them
//                  and the boxes each line completed.
//   game -> agent  {"type":"error","msg":"..."}  (the command is ignored)
//
// Computer players move with a 0 ms timer instead of 200 ms and the score
// dialogs are skipped. The game quits when the socket closes.

#include <QByteArray>
#include <QJsonArray>
#include <QObject>

class KSquaresWindow;
class KSquaresGame;
class QSocketNotifier;

class RLBridge : public QObject
{
    Q_OBJECT

public:
    /// Connects if KSQUARES_RL_SOCKET is set; returns nullptr otherwise.
    static RLBridge *start(KSquaresWindow *window, KSquaresGame *game);

    /// An external (human-seat) player is to move.
    void awaitMove();
    /// The game is over.
    void gameOver();
    /// Delay before a computer move in ms: KSQUARES_RL_AI_DELAY_MS, default 0
    /// (the game itself waits 200 ms; 0 for training, more for watching).
    static int aiDelayMs();

private:
    RLBridge(KSquaresWindow *window, KSquaresGame *game, int fd);
    void onReadable();
    void handle(const QByteArray &line);
    void sendJson(const QByteArray &json);
    void sendError(const QString &msg);
    void sendState(bool done);

    KSquaresWindow *m_window;
    KSquaresGame *m_game;
    int m_fd;
    QSocketNotifier *m_notifier;
    QByteArray m_inbuf;
    QJsonArray m_moves;    // lines drawn since the last state message
    bool m_pending = false; // a reset/step waits for its state message
    bool m_waitingMove = false;
};

#endif // RLBRIDGE_H
