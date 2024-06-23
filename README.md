
# Implémentation du Jeu de Grille Hexagonale
Alexandre EBERHARTD, Oriane LANFRANCHI

Ce projet implémente les jeux "Gopher" et "Dodo", conçus par Mark Steere, sur une grille hexagonale. Les jeux impliquent deux joueurs, Rouge et Bleu, qui placent des pierres ou déplacent des pièces sur une grille hexagonale initialement vide. Les objectifs varient selon les règles spécifiques de chaque jeu.

## Structure du Projet

### Fichiers et Répertoires

-   `Main.py` : Le script principal.
-   `Gopher.py`: Le script principal pour exécuter la simulation du jeu Gopher.
-   `Dodo.py`: Le script principal pour exécuter la simulation du jeu Dodo.
-   `Grid.py`: Contient des fonctions utilitaires et des classes pour gérer la grille hexagonale, la gestion des états, et la logique du jeu.
-   `README.md`: Ce fichier, fournissant une vue d'ensemble et des instructions pour le projet.
-   `Gopher_hex_rules.pdf`: Les règles officielles du jeu "Gopher" de Mark Steere.
-   `Dodo_rules.pdf`: Les règles officielles du jeu "Dodo" de Mark Steere.


## Explication du Code

### Mémorisation

Deux décorateurs de mémorisation, `memoize` et `memoize_ab`, sont utilisés pour mettre en cache les résultats des algorithmes minimax et negamax afin d'optimiser les performances.

### Initialisation de la Grille

-   `initGrid(size: int) -> Grid.Grid`: Initialise une grille hexagonale vide de la taille donnée.

### Actions Légales

-   `legals(grid: Grid.Grid, player: Grid.Player) -> list[Grid.ActionGopher]`: Détermine les actions légales disponibles pour un joueur sur la grille actuelle.
-   `evalCellsAround(rowAx: int, colAx: int, grid: Grid.Grid, player: Grid.Player, opponent: Grid.Player) -> bool`: Vérifie si une cellule est jouable par le joueur donné.

### Logique des Jeux
#### Gopher
-   `isFinal(grid: Grid.Grid, currentPlayer: Grid.Player) -> bool`: Vérifie si le jeu est dans un état final (c'est-à-dire, plus de mouvements légaux pour le joueur actuel).
-   `score(state: Grid.State, currentPlayer) -> float`: Évalue le score de l'état actuel pour le joueur donné.
-   `play(state: Grid.State, action: Grid.ActionGopher, currentPlayer: Grid.Player) -> Grid.State`: Exécute un mouvement et retourne le nouvel état.
-   `strategyUser`: Permet à un joueur humain de choisir un mouvement parmi les actions légales.
-   `strategyMinimax`: Utilise l'algorithme minimax pour choisir le meilleur mouvement.
-    `strategyAlphaBeta`: Utilise l'algorithme minmax avec élagage alpha-bêta pour choisir le meilleur mouvement.
-    `strategyNegaMax`: Utilise l'algorithme negamax pour choisir le meilleur mouvement.
-   `strategyNegaMaxAlphaBeta`: Utilise l'algorithme negamax avec élagage alpha-bêta pour choisir le meilleur mouvement.
-   `strategyRandom`: Choisit un mouvement aléatoirement parmi les actions légales.

#### Dodo

- `isFinal(grid: Grid.Grid) -> bool`: Vérifie si le jeu est dans un état final (aucun mouvement légal pour l'un des joueurs).
- `isFinalPlayer(grid: Grid.Grid, player: Grid.Player) -> bool`: Vérifie si un joueur donné n'a plus de mouvements légaux.
- `score(state: Grid.State) -> float`: Évalue le score de l'état actuel pour le jeu Dodo.
- `strategyUser(state: Grid.State, player: Grid.Player) -> Grid.ActionDodo`: Permet à un joueur humain de choisir un mouvement parmi les actions légales.
- `strategyRandom(env: Grid.Environment, state: Grid.State, player: Grid.Player, time_left: Grid.Time) -> tuple[Grid.Environment, int, Grid.Action]`: Choisit un mouvement aléatoirement parmi les actions légales.
- `strategyAlphaBeta(env: Grid.Environment, state: Grid.State, player: Grid.Player, time_left: Grid.Time) -> tuple[Grid.Environment, int, Grid.Action]`: Utilise l'algorithme alpha-bêta pour choisir le meilleur mouvement.
- `strategyNegaMax(env: Grid.Environment, state: Grid.State, player: Grid.Player, time_left: Grid.Time, depth: int = 3) -> Tuple[Grid.Environment, Grid.ActionDodo]`: Utilise l'algorithme negamax pour choisir le meilleur mouvement.

### Fonctions d'Évaluation

- `evaluation(state: Grid.State, player: Grid.Player) -> float`: Évalue l'état du jeu pour le joueur donné en utilisant la différence entre le nombre de coup légaux du joueur et de son opposant.
- `numberOfLegals(state: Grid.State, player: Grid.Player) -> float`: Retourne le nombre de mouvements légaux restants pour un joueur donné.
- `blockedPieces(state: Grid.State, player: Grid.Player) -> float`: Retourne le nombre de pièces d'un joueur donné bloquées pour ce tour.
- `burriedPieces(state: Grid.State, player: Grid.Player) -> float`: Retourne le nombre de pièces enterrées d'un joueur donné (qui ne peuvent plus être déplacées).

### Fonction Principale

-   `main()`: Exécute une série de jeux et imprime les résultats, en comparant les performances de différentes stratégies.
- `gamePlay(size:  int,  strategyRed: Grid.Strategy,  strategyBlue: Grid.Strategy)` : Simule une partie avec des appels aux stratégies du joueur rouge et du joueur bleu, sur une grille de taille spécifiée.

### Fonctions d'Affichage

-   `printGridFlat(grid: Grid)`: Affiche la grille orientée avec le sommet plat.
-   `printGridPointy(grid: Grid)`: Affiche la grille orientée avec le sommet pointu.

## Comment Exécuter

Pour exécuter la simulation des jeux, lancez le script `Gopher.py` ou `Dodo.py` :

`python Gopher.py` 
ou 
`python Dodo.py` 

Cela simulera X parties entre les stratégies choisies, et affichera les résultats en nombre de parties gagnées par le joueur 1 / nombre de parties totales.

## Règles de Gopher

Consultez le document `Gopher_hex_rules.pdf` pour les règles officielles du jeu. Voici un bref aperçu :

-   **Objectif**: Le dernier joueur à placer une pierre gagne.
-   **Ordre des Tours**: Rouge place la première pierre. Ensuite, Bleu et Rouge alternent leurs tours.
-   **Règles de Placement**: Les pierres doivent être placées sur des cellules inoccupées. Chaque pierre doit former exactement une connexion ennemie et aucune connexion amicale.

Pour des règles détaillées, veuillez vous référer au document `Gopher_hex_rules.pdf` inclus dans le projet.

## Règles de Dodo

Consultez le document `Dodo_rules.pdf` pour les règles officielles du jeu. Voici un bref aperçu :

- **Objectif**: Si, au début de votre tour, vous n'avez aucun mouvement disponible, vous gagnez.
- **Ordre des Tours**: Rouge commence par déplacer une de ses pièces. Ensuite, Bleu et Rouge alternent leurs tours.
- **Règles de Déplacement**: Les pièces doivent être déplacées sur des cellules inoccupées. Les joueurs peuvent déplacer leurs pièces d'une cellule directement en avant ou en diagonale avant.

Pour des règles détaillées, veuillez vous référer aux documents `Gopher_hex_rules.pdf` et `Dodo_rules.pdf` inclus dans le projet.
## Auteur

Ce projet a été inspiré par les jeux "Gopher" et "Dodo" de Mark Steere. La logique du jeu et les stratégies ont été implémentées dans le cadre du cours d'IA02 à l'UTC.
# Réalisation du projet
Nous abordons, dans cette partie, les différentes pistes et stratégies adoptées afin de mener à bien ce projet. Nous nous attardons en particulier sur les stratégies de jeu des IA.

Afin d'optimiser les stratégies réalisées, nous avons fait jouer nos IA contre une strategy random (c'est-à-dire choisissant un coup legal au hasard), mais nous avons également et en majorité organisé des parties avec d'autres binômes d'IA02, afin d'évaluer les performances de nos programmes respectifs, et les ajuster en conséquence.

## Gopher
Dans a version finale du projet, la stratégie adoptée par l'IA dans le jeu Gopher repose sur l'utilisation d'un algorithme NegaMax avec un élagage alpha-bêta, couplée à l'utilisation d'une fonction d'évaluation. La recherche NegaMax est une variante de la recherche MinMax.

Nous avions, dans un premier temps, mis en place-ce dernier. Le nombre de possibilités de coups étant trop élevée pour réaliser une recherche MinMax complète, nous l'avons, dans un premier temps, accélérée avec un élagage alpha-bêta, puis limitée en profondeur en lui associant une fonction d'évaluation. Nous avons établi une fonction d'évaluation basée sur le nombre de coups légaux du joueur courant par rapport au nombre de coups légaux du joueur adverse, et permettant de valoriser les états où le joueur courant possède plus de coups légaux disponibles que son adversaire ; en effet, l'objectif du jeu Gopher est d'être le dernier joueur à ne plus avoir de coups légaux.

En parallèle, nous avons réalisé une recherche MinMax avec mise en place d'un cache, afin de comparer les temps d'exécution des deux stratégies. L'objectif était de mettre en cache le résultat de chaque état, ainsi que les symmétries et rotations de cet état. Toutefois, le calcul de ces-derniers ne permettait pas de bénéficier d'un gain de temps sur l'utilisation du cache.

Nous avons donc combiné le MinMax avec élagage alpha-bêta et la mise en cache des états évalués (mais pas leurs symmétries ni rotations). Toutefois, le temps d'exécution de la recherche MinMax était toujours trop importante. Nous avons donc cherché d'autres méthodes de recherche de coup optimal. C'est ainsi que nous avons découvert et implémenté la recherche en NégaMax, qui montre une exécution plus rapide que la recherche en MinMax.

Nous avons ajouté un élagage alpha-bêta afin d'augmenter ses performances, mais pas conservé la mise en cache des états évalués. En effet : si le cache permet d'effectuer une recherche plus rapide, son utilisation en parallèle d'un élagage alpha-bêta baisse le taux de victoire de l'IA. L'élgage alpha-bêta renvoie effectivement une estimation d'un état de jeu ; mettre en cache ces données peut renvoyer à une évaluation erronée au cours de la partie. Nous avons constaté que la perte de victoire avec utilisation du cache était, sur un set de 100 parties, d'environ 5%. 

Cette méthode permet de réaliser une partie avec le joueur IA en joueur rouge (joueur 1) et un joueur random en joueur bleu (joueur 2) en environ 0.5 secondes sur une grille hexagonale de taille 5. Sur 100 parties lancées, environ 95 sont remportées par le joueur IA.

Dans le cas d'une partie sur un plateau hexagonal de taille 4, avec une profondeur de recherche infinie (c'est-à-dire que la recherche NegaMax évalue l'ensemble des états finaux), le temps de jeu de la partie complète est d'environ 22 secondes.

## Dodo
Dans la version finale du projet, la stratégie adoptée par l'IA dans le jeu Dodo se base sur une méthode Monte Carlo.

Nous avions, dans un premier temps, implémenté une recherche de coup optimal avec MinMax, accélérée par un élagage alpha-bêta. Le nombre de coups possibles dans le jeu Dodo étant largement supérieurs à ceux du jeu Gopher, nous avons immédiatement jugé nécessaire la mise en place d'une fonction d'évaluation, afin de fixer la profondeur de recherche de MinMax. Pour un gain de temps, nous avons ensuite préféré l'utilisation de la recherche en NegaMax.

La fonction d'évaluation réalisée estime le nombre de coups légaux du joueur courant en rapport à celui de son adversaire, favorisant, cette fois, un nombre de coups inférieur de coups légaux par rapoport à l'adversaire, l'objectif du jeu Dodo étant d'être le premier joueur à ne plus pouvoir jouer.

Nous avions également mis en place une autre fonction d'évaluation, basée sur le concept de *race turns left*. Cette technique consiste à imaginer le plateau sans les pions adverses, et compter le nombre de tours minimums avant que ses propres pions soient bloqués contre le coin adverse - et donc de ne plus pouvoir jouer. La différence du nombre de *race turns left* d'un joueur et de celui de son adversaire permet de donner une indication sur quel joueur a - d'apparence - l'avantage dans le jeu.

Toutefois, cette technique implique le calcul répétitif de coups légaux afin de déterminer l'itinéraire le plus court. De plus, il s'agit d'une pure estimation, qui peut se révéler fausse, selon le placement des pions sur le plateau de jeu (pions bloquant définitivement d'autres pions, par exemple). Le temps d'exécution de cette fonction était finalement trop important pour le résultat apporté ; nous avons décidé de conserver la fonction d'évaluation se basant sur la différence des coups légaux entre les joueurs.

Nous avons également testé une autre stratégie (dite stratégie mix), basée sur le concept du *race turns left*, c'est-à dire le fait de miser sur l'avancée des pions vers le coin adverse du plateau. Cette stratégie privilégie les coups permettant d'avancer ses pions tout droit.

Nous avons également testé la mise en place d'un cache. En comparant la vitesse d'exécution sur plusieurs parties, avec et sans cache, nous avon spu constater que l'utilisation du cache permettait une exécution plus rapide. Toutefois, nous avons pu constater que le taux de victoire n'est pas amélioré (et même décru) par l'utilsiation du cache. Cela peut s'expliquer par l'utilisation de l'élagage alpha-bêta, qui réalise une estimation des états, et ne renvoie pas une image exacte des possibilités du jeu. 

Nous avons ainsi abandonné l'idée d'utiliser un cache. De plus, nous avons décidé de tester une recherche de coup optimal en implémentant une méthode Monte Carlo, plus efficace pour une grande multiplicité de cas.

La méthode actuellement implémentée sur Dodo permet d'exécuter une partie complète (joueur 1 utilisant la stratégie Monte Carlo, joueur 2 utilisant une stratégie random) en environ 8 secondes, avec 100% de victoire contre un joueur choisissant ses coups de manière aléatoire (random).

# Lisibilité du code
Afin d'assurer un code proprement rédigé, nous avons utilisé pylint, black.

# Lancer le programme

Il y a plusieurs possibilités pour lancer le programme.

Il est possible de le faire tourner de manière locale, ou bien directement sur le serveur de M. Lagrue.

## Pour lancer le programme en local
Selon le jeu souhaité, il faut se rendre dans le fichier gopher.py ou dodo.py.

Le procédé est similaire pour le fichier gopher.py et dodo.py.
Rendez-vous dans la fonction main() du programme. Y est présent un appel à la fonction game_play(). Cette fonction implémente une boucle de jeu sur une partie, et retourne le score de la partie (1 si le joueur rouge gagne, -1 si le joueur bleu gagne). Elle est de la forme suivante :

````
game_play(size: int, strategy_red: Grid.Strategy, strategy_blue: Grid.Strategy) -> float
````

Le paramètre *size* représente la taille du plateau de jeu.
Le paramètre  *strategy_red* représente la stratégie à utiliser pour le joueur rouge.
Le paramètre *strategy_blue* représente la stratégie à utiliser pour le joueur bleu.

N'hésitez pas à faire varier ces paramètres afin de comparer les différentes stratégies que nous avons mis en place. Une liste (non-exhaustive) des stratégies de chaque jeu est disponible plus en aval dans ce document. Par défaut la stratégie utilisée par le joueur rouge est la stratégie finale de notre projet, que nous avons utilisée lors de la compétition d'IA du vendredi 21 juin 2024. Par défaut, la stratégie utilisée pour le joueur bleu est une stratégie choisissant un coup au hasard (stratégie random).

Nous avons mis en place un système d'affichage de la grille au fur et à mesure de la partie, dès qu'un joueur joue un coup.

Pour lancer le programme, il suffit simplement d'exécuter le fichier gopher.py ou dodo.py dans votre terminal.

Pour gopher, la commande suivante dans un terminal :
````
python gopher.py
````

Pour dodo, la commande suivante dans un terminal :
````
python dodo.py
````

## Pour lancer le programme sur le serveur
Il faut de se rendre dans le fichier main.py.

L'URL du serveur est par défaut "http://lagrue.ninja/".

Les stratégies appelées par défaut pour jouer au jeux gopher et dodo se trouvent dans la fonction *strategy_brain*. La stratégie appelée par défaut pour gopher est :
````
gopher.strategy_nega_max_alpha_beta
````

La stratégie appelée par défaut pour dodo est :
````
dodo.strategy_mc
````

Vous êtes libres de faire varier ces paramètres afin de tester les différentes stratégies que nous avons mises en place. Une liste des stratégies réalisées, ainsi que le nom des fonctions associées, est disponible plus en aval dans ce document.

Afin d'exécuter le programe il suffit d'utiliser la commande suivante sur terminal :

````
main.py 1 Groupe Mdp -jeu
````

## Stratégies Gopher
Une liste non-exhaustive des stratégies mises en place pour gopher. Nous avons choisi de faire paraître uniquement les stratégies que nous avons jugées les plus intéressantes. (Un Ctrl+F dans le fichier gopher.py, en recherchant "strategy" permet de trouver toutes les stratégies présentes dans le fichier)

(la vraie raison pour laquelle la liste n'est pas exhaustive est peut-être également car ce readme est déjà **très** long)

### Stratégie NegaMax avec élagage alpha-bêta
````
gopher.strategy_nega_max_alpha_beta
````

Au début du projet, nous avions utilisé un algorithme MinMax, sans puis avec élagage alpha-bêta. Dans un soucis de performances et en particulier de rapidité d'exécution, nous avons décidé d'utiliser à la place un algorithme NegaMax, pourvu d'un élagage alpha-bêta, en profondeur finie. La profondeur que nous avons fixée est, par défaut, 4. Vous êtes libres de faire varier ce paramètre si vous souhaitez tester la stratégie.

Puisque nous nous situons en profondeur finie, nous avons recours à une fonction d'évaluation, qui jauge l'état du plateau de jeu en effectuant un rapport du nombre de coups légaux de l'adversaire par rapport au nombre de nos coups légaux.

### Stratégie NegaMax (avec cache)
````
gopher.strategy_nega_max
````

En étudiant les possibilités autour de l'algorithme NegaMax, nous avons eu l'idée d'utiliser un cache, en utilisant la méthode memoize vue en cours magistral. Nous avons fixé une profondeur de recherche à 3, et avons donc utilisé une fonction d'évaluation, toujours basée sur le rapport de coups légaux entre les joueurs. Nous avions, dans un premier temps, enregistré les symmétries et rotations du plateau dans le cache. Cependant, nous nous sommes rendus compte que le temps de calcul de ces symmétries et rotation ne permettait pas de bénéficier d'un gain de temps suffisant par rapport à l'utilisation d'un cache - au contraire, nous avons remarqué une perte de temps. Nous avons donc abandonné l'idée d'enregistrer les symmétries et rotations dans le cache (les vestiges des fonctions permettant de calculer ces états se situent toujours dans le fichier grid.py).

### Stratégie ChatGPT
````
gopher.strategy_chatgpt
````

Cette stratégie demande à chatGPT 3.5 quel coup jouer. Nous avons testé une stratégie en utilisant l'API de GPT 3.5. Il s'agit d'une experience plus que d'une véritable statégie gagnante (impliquant le fait qu'elle soit traitée dans une partie à part de ce document), mais elle fonctionne et gagne environ 80% du temps contre un joueur random sur gopher.

Elle pourrait être implémentée dans dodo sans difficulté supplémentaire.

Lors de l'appel à la stratégie, nous commencons par determiner les coups légaux, puis nous envoyons une requête API via openai pour le modèle GPT 3.5 turbo (le modèle le plus rapide disponible).
voici le prompt utilisé :

messages=[
            {"role": "system", "content": '''You are a helpful assistant
 with extensive knowledge of gopher. '''},
            {"role": "user", "content": f'''Given the following position, you are
player {player} on the board {state}, what move should I play next, here are the
legals plays :{list_actions}.? Give only your choice like this format, without any
additional text, or else it will break my game: (x,y)'''}
        ]

Le role permet de spécifier le role de chatgpt, un assistant de jeu pour le gopher.

Ensuite, un prompt lui est transmis lui donnant les information de jeu : le joueur qu'il est, l'état du jeu, et les coups légaux à sa disposition.

Il faut finalement préciser enfin qu'il doit chosir un coup légal et le format de retour, pour ne pas avoir de problème pour traiter la réponse. Un output avec un format défini est bien plus simple à parser.

Après plusieurs essais, cela fonctionne, le nombre de coups illégaux est relativement faible, le temps de réponse est très bon.

Il gagne la plupart du temps contre un joueur random et a pu gagner quelques fois contre d'autres étudiants.

Perspectives :
- Nous avons essayé de lui donner les règles du jeu (qui sont toujours dans le code) mais ces informations ne l'aident pas et un trop long prompt augmentait la proportion de coups illégaux.
- Il est envisageable de tester d'autres versions de chatgpt accessibles via requetes api, le temps de réponse étant régulier, un modèle plus lent pourrait fonctionner.

Il faut garder en tête que l'API est payante ; le coût à la requête étant marginal, nous vous invitons à l'essayer, mais merci de limiter la taille des grilles à 6.
Voilà une clé d'API avec des crédits, à remplacer dans la chaine "Your API key" :
"XXXXX"
Merci de ne pas la partager en dehors du cours d'IA02 et de ne pas en faire une utilisation excessive.

## Stratégies Dodo
 Une liste non-exhaustive des stratégies mises en place pour dodo. Nous avons choisi de faire paraître uniquement les stratégies que nous avons jugées les plus intéressantes. (Un Ctrl+F dans le fichier dodo.py, en recherchant "strategy" permet de trouver toutes les stratégies présentes dans le fichier)

(la vraie raison pour laquelle la liste n'est pas exhaustive est peut-être également car ce readme est toujours **très** long)
### Stratégie MC
````
dodo.strategy_mc
````

La stratégie que nous avons finalement choisi pour le dodo est un algorithme de Monté Carlo (MC). La très grande diversité d'actions légales au dodo nous a poussé à considérer que les algorithme "par étages" comme le mimax/negamax, allant jusqu'a une certaine profondeur, n'étaient pas optimal. En effet : le nombre de possibilités (nombre de coup légaux) est puissance(profondeur).

Nous avons donc opté pour un algorithme MC, qui explore des branches par itération, mais qui est certain d'aller au bout.

Cet algorithme avait une bien meilleure effciacitée, et ne nécessitait pas de fonction d'évaluation.

Dans les perspectives de cette stratégie, nous pourrions l'optimiser pour augmenter le nombre d'itération, et tester sur suffisamment de parties afin de connaître le nombre d'itérations maximal avant de rencontrer un timeout. Nous pourrions aussi décider du nombre d'itérations selon le temps de jeu restant au joueur, en attribuant le nombre d'itérations en fonction de ce-dernier afin de l'utiliser au maximum.

### Strategie mix

````
dodo.strategy_mix
````

Dans un premier temps, nous avons cherché à optimiser l'algorithme NegaMax pour aller plus profond dans la recherche de coup otpimal. Nous avons eu l'idée d'une stratégie basée sur le concept du *race turns left*, c'est-à dire le fait de miser sur l'avancée des pions vers le coin adverse du plateau ; ce concept implique un déplacement des pions vers l'avant le plus possible. Il s'agit d'une représentation irréaliste du jeu, puisque les pions de l'adversaire son ignorés, et que la seule stratégie prise en compte est celle de bloquer ses pions au niveau du coin adverse.

En se basant sur ce principe, nous avons, un premier temps, nous avons donc implémenté la stratégie forward, qui permet de jouer les coups permettant d'avancer ses pions en ligne droite au maximum. Si aucun pion ne peut être avancé en ligne droite, la stratégie chosiit de jouer le premier coup légal. Evidemment, la stratégie forward n'est pas optimale.

Nous avons donc eu l'idée de mélanger cette stratégie avec l'utilisation du NegaMax.

Nous avons alors cherché combien de coup nous pouvions jouer tout droit sans perdre (cf. le graphique.png qui présente nos résultats). Nous pouvons donc jouer 10 coups sans perdre, avant d'arriver au milieu de partie, où nous essayons de bloquer nos pions. C'est à partir de ce moment que nous appelons l'algorithme NegaMax.

Ensuite, il est envisageable de rappeler l'algorithme forward s'il n'y a pas de blocage au milieu de la partie. 

Ainsi, nous sommes supposés être plus rapide pour atteindre le fond du plateau et bloquer les pions.

Cette stratégie, bien que fonctionnelle, n'était pas aussi efficace qu'espérée, c'est pourquoi nous ne l'avons pas retenue dans notre version finale du projet.

### Strategie NegaMax avec élagage alpha-bêta
````
dodo.strategy_nega_max_alpha_beta
````

Nous avons tenté d'utiliser la méthode NegaMax pour le jeu dodo, avec un élagage alpha-bêta afin d'améliorer les performances en particulier en terme de temps - il s'agit d'une amélioration de la méthode MinMax avec élagage alpha-bêta que nous avions utilisé dans un premier temps. Ce n'était pas une solution optimale, mais la stratégie est toujours disponible. Nous avons effectué une recherche en profondeur finie (profondeur 6 par défaut), avec une fonction d'évaluation jaugeant le plateau en comparant le nombre de coups des deux joueurs.

### Strategie NegaMax
````
dodo.strategy_nega_max
````

Il s'agit de l'ancêtre de la stratégie précédente, sans élagage alpha-bêta. La recherche de coup optimal s'effectue en profondeur finie (par défaut, 3), avec une fonction d'évaluation jaugeant le plateau en comparant le nombre de coups des deux joueurs.