"""Language of the part 3 materials: P3_LANG=en selects the English cut.
T() maps a Russian screen text to the chosen language; file names get suffix()."""
import os
import re

LANG = os.environ.get("P3_LANG", "ru")
STRINGS = {
    " (заморожена)": " (frozen)", "% верных ходов": "% right moves", ",": ".",
    "10 из 17: один ход от выигрыша к проигрышу, 7–20 безопасных линий": "10 of 17: one move from win to loss, at 7–20 safe lines",
    "100 % верно": "100% exact", "17 разгромов: где судья нашёл ошибку": "17 routs: where the judge found the mistake",
    "2 из 24": "2 of 24", "6 из 24": "6 of 24",
    "200 вариантов: 96 % верных ходов там, где судья может проверить": "200 options: 96% right moves where the judge can check",
    "Dot-2 без размышлений": "Dot-2 without thinking", "Dot-2 с размышлениями": "Dot-2 with thinking",
    "Большой набор судьи: 1925 позиций": "The judge's big set: 1925 positions",
    "Думая, Dot-2 считает,\nкому открывать первую цепочку.\nБез размышлений она этого не видит.":
        "When she thinks, Dot-2 counts\nwho has to open the first chain.\nWithout thinking she doesn't see it.",
    "Заморозили интуицию": "Intuition frozen", "Зелёная —": "Green is",
    "Знакомые ошибки Dot-2 запоминает, а новые?": "Dot-2 memorizes known mistakes. And new ones?",
    "И белая": "And white", "Конец партии: доска разрезана на цепочки": "End of the game: the board is cut into chains",
    "Концовка": "The endgame", "Кто что знает": "Who knows what", "Начало партии": "The start of the game",
    "Наш вопрос": "Our question", "Нейросеть Dot, часть 3": "Neural network Dot, part 3", "Окошко или граф": "Window or graph",
    "Оранжевая": "Orange is", "По вертикали": "Vertically", "По горизонтали": "Horizontally",
    "Понять, а не пересчитать": "Understand, not recount", "Попытка 1: Dot-2 учится у судьи": "Attempt 1: Dot-2 learns from the judge",
    "Попытка 2: взгляд на всё поле и подсказка": "Attempt 2: whole-board view and a hint",
    "Попытка 3: судья по формуле": "Attempt 3: a judge by formula", "Попытка 4: сильный судья": "Attempt 4: a strong judge",
    "Попытка 5: на своих ошибках": "Attempt 5: her own mistakes", "Поэтому в коробке": "That's why the box",
    "Середина партии": "The middle of the game", "Синяя": "Blue is",
    "Стена: сколько позиций судья решает": "The wall: how many positions the judge solves",
    "Судья видит только конец партии": "The judge sees only the end of the game", "Что видит сеть": "What the network sees",
    "без размышлений": "without thinking", "без размышлений: 67 → 75 %": "without thinking: 67 → 75%",
    "без размышлений: 67 → 83 %": "without thinking: 67 → 83%", "безопасных линий": "safe lines",
    "безопасных линий осталось": "safe lines left",
    "безопасных линий\nне осталось,\nдоска разрезана\nна цепочки:": "no safe lines\nleft, the board\nis cut\ninto chains:",
    "в остальных 7 партия проиграна ещё раньше: 13–29 безопасных линий": "in the other 7 the game was lost earlier: 13–29 safe lines",
    "в середине партии\nдоска ещё не разрезана:\nсчитать нечего": "in the middle of the game\nthe board isn't cut yet:\nnothing to count",
    "вариантов:": "options:", "верные ходы без размышлений": "right moves without thinking", "взгляд на всё поле": "view of the whole board",
    "взглянула и сразу ходит": "one look and she moves", "видит проигрыш там, где его нет": "sees losses where there are none",
    "всё решает,\nкому придётся\nоткрыть первую": "it all depends\non who has to\nopen the first",
    "выбор хода": "move choice", "выигрыш": "win", "графовая сеть: цепочки готовыми": "graph network: chains ready-made",
    "за 10 минут": "in 10 minutes", "за минуту": "in a minute", "здесь ошибка": "the mistake is here", "знакомые ошибки": "known mistakes",
    "игроки: до 18 линий, дальше мало позиций для проверки": "players: up to 18 lines, beyond that too few positions to check",
    "интуиция": "intuition", "интуиция: кто выигрывает": "intuition: who is winning",
    "концовка\nот 20 линий до конца": "endgame\nfrom 20 lines to the end", "линия": "line",
    "настоящие ошибки самой Dot-2": "Dot-2's own real mistakes", "начало партии": "start of the game", "не достаёт": "can't reach",
    "новые ошибки": "new mistakes", "общее тело": "shared body", "ответов нет": "no answers", "ошибка найдена": "mistake found",
    "ошибок в середине": "mid-game mistakes", "партий": "games", "партия": "game", "первая линия": "first line",
    "перебор на настоящей доске, на листьях формула": "search on the real board, the formula at the leaves",
    "подсказка: цепочки,\nкому открывать первую": "hint: the chains,\nwho opens the first",
    "подсказка:\nкакие цепочки,\nкому открывать первую": "hint:\nwhich chains,\nwho opens the first",
    "позиции из партий Dot-2: 20 на лимит в минуту, 8 на лимит в 10 минут": "positions from Dot-2's games: 20 for the minute limit, 8 for the 10-minute limit",
    "понимать игру, а не пересчитывать её каждый раз?": "understand the game, not recount it every time?",
    "порог, заданный заранее,\nне пройден\n(партии не игрались)": "the threshold set in advance\nwas not passed\n(no games were played)",
    "последняя линия": "last line", "прибавка есть, доказать её не смогли": "there is a gain, but we couldn't prove it",
    "проигрыш": "loss", "пять попыток научить Dot-2 видеть правильный ход сразу": "five attempts to teach Dot-2 to see the right move at once",
    "разгромов": "routs", "разгромов 11 → 46 из 200  ·  настоящий противник 19–0–1 → 15–5": "routs 11 → 46 of 200  ·  real opponent 19–0–1 → 15–5",
    "разгромов 12 из 200  ·  настоящий противник 19–1: игра цела, разгромы те же": "routs 12 of 200  ·  real opponent 19–1: game intact, same routs",
    "разгромов из 200": "routs of 200", "разгромы те же": "the same routs", "решающие позиции из новых партий": "decisive positions from new games",
    "с размышлениями": "with thinking", "свёрточная сеть: окошко": "convolutional network: a window",
    "свёрточное окошко:\nмаленькие узоры": "convolutional window:\nsmall patterns",
    "середина\n20–40 безопасных линий": "middle\n20–40 safe lines", "стало верно: 11": "became right: 11", "стало неверно: 4": "became wrong: 4",
    "судья": "judge", "судья молчит": "the judge is silent", "судья не дотянулся": "judge couldn't reach", "судья отвечает": "the judge answers",
    "судья: доля решённых": "judge: share solved", "так мало, что от шума не отличить": "too few to tell from noise",
    "цепочка длиннее окошка": "the chain is longer than the window",
    "часть 4: графовая сеть вместо свёрточной": "part 4: a graph network instead of a convolutional one",
    "Что знает нейросеть —": "What a neural network knows —", "и где кончается знание": "and where knowing ends",
}


def T(s):
    if LANG == "ru":
        return s
    if re.fullmatch(r"[\d.,]+ %", s):
        return s.replace(" ", "").replace(",", ".")
    if re.fullmatch(r"[\d.]+", s):
        return s
    return STRINGS[s]


def suffix():
    return "" if LANG == "ru" else f"_{LANG}"
