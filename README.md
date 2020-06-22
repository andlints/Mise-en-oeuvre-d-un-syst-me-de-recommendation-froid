# Mise en oeuvre d'un système de recommendation à froid

# ÉTUDE BIBLIIOGRAPHIQUE

## Quelques définitions:

* Les systèmes de recommandation sont une forme spécifique de filtrage de l’information (SI) visant à présenter les éléments d’information susceptibles d’intéresser l’utilisateur. Généralement, un système de recommandation permet de comparer le profil d’un utilisateur à certaines caractéristiques de référence, et cherche à prédire l’«avis» que donnerait un utilisateur. Ces caractéristiques peuvent provenir de:

	- l’objet lui-même, on parle «d’approche basée sur le contenu» ou «content-based approach»
	- l’utilisateur
	- l’environnement social, on parle «d’approche de filtrage collaboratif» ou «collaborative filtering»

* Cold-start (recommander systèms) ou démarrage à froid : c’est un phénomène qui se produit en début d’utilisation du système, dans des situations critiques où le système manque de données pour procéder à un filtrage personnalisé de bonne qualité. 
On distingue 3 types de problèmes de démarrage à froid (Burke, 2002):
	- le démarrage à froid pour un nouveau système (‘new system’): les performances des systèmes sont très mauvaises  en raison de l’absence d’informations sur lesquelles fonder le processus de filtrage personnalisé.
	- le démarrage à froid pour un nouveau document (‘new item’): c’est un problème spécifique de l’approche collaborative, pour laquelle les objets à recommander ne sont décrits que par les évaluations fournies par les utilisateurs. Ce problème est généralement traité en combinant une approche de filtrage basé sur le contenu avec l’approche collaborative (approche hybride) par exemple utilisant la similarité entre documents (Schein et al., 2001) ou introduisant des agents intelligents qui évaluent les documents automatiquement ( Good et al.; 1999)
	- le démarrage à froid pour un nouvel utilisateur (‘new user’): le profil de l’utilisateur est inexistant et ses communautés sont encore inconnues, ce qui conduit à des recommandations de mauvaise qualité.

* Les systèmes affectés:
Le problème de démarrage à froid est un problème bien connu et bien étudié pour les systèmes de recommandation. Les systèmes de recommandation forment un type spécifique de technique de filtrage de l’information (IF) qui tente de présenter des informations (e-commerce, films, music, books, news, images, web pages) susceptibles d’intéresser l’utilisateur. En règle générale, un système de recommandation compare le profil de l’utilisateur à certaines caractéristiques de référence. Ces caractéristiques peuvent être liées aux caractéristiques des éléments (filtrage basé sur le contenu)  ou à l’environnement social de l’utilisateur et à son comportement antérieur (filtrage collaboratif). Selon le système, l’utilisateur peut être associé à différents types d’interactions comme les scores, les signets, les likes, le nombre de visites de page, etc....

Dans le cadre de ce stage, on va plus s’intéresser aux problèmes liés à une approche de filtrage collaboratif (CF). C’est à dire, le démarrage à froid pour un nouveau document. Afin de résoudre cela, on va étudier les méthodes proposées dans les articles suivants:





# ARTICLE 1: 
# «Wasserstein Collaborative Filtering for Item Cold-start Recommandation»  

Comme le titre l’indique, nous allons nous placer dans le problème du démarrage à froid pour un nouveau document.
En effet, l’ajout de nouveau document  ou le manque, voir très peu d’interaction limite la performance d’une approche collaborative. Afin de résoudre ce problème, plusieurs méthodes ont été mise en place, comme la prédiction de l’interaction du nouveau document à partir du contenu.
Cependant, on est confronté à une difficulté à concevoir et à appendre une carte entre l’historique d’interaction de l’élément et le contenu correspondant.
L’article propose ainsi, l’utilisation de la distance de Wasserstein pour traiter le problème de démarrage à froid.  Sur un point de vue général, compte tenu des informations présentes dans les éléments, nous allons pouvoir calculer la similarité entre les éléments existant (qui interagissent déjà entre eux), et les éléments nouveaux du démarrage à froid, de sorte à ce que la préférence de l’utilisateur sur ces derniers puisse être déduite par la distance de Wasserstein entre la distribution de ces deux types d’éléments.
Nous adoptons en outre l’idée du filtrage collaboratif (CF) et proposons Wasserstein CF (WCF) pour améliorer les performances de recommandation sur les éléments de démarrage à froid.

## 1 - Introduction

En raison de la quantité très importante d’informations sur Internet. Les systèmes de recommandation ont pris une place primordiale dans le but de faire des suggestions aux utilisateurs dans leurs démarches. Le CF a été la technologie de recommandation à adopter en raison de ses performances prometteuses qui ont été démontrées lors de la coupe KDD [G. Dror et Weimer, 2012] et la compétition Netflix [Bell et Koren, 2007].  Pour un utilisateur spécifié, CF recommande des articles selon la préférences des utilisateurs ayant un historique similaire. Cependant, CF est incapable de traiter les nouveaux éléments sans historiques d’interaction pertinent. D’où le problème de démarrage à froid.
	
	Études récentes:  [Saveki et Mantrach, 2014; Iman Barjasteh et Esfahanian, 2016]

Ces études montrent que le problème de démarrage à froid pour un élément peut être efficacement atténué en prenant en compte les informations contenus dans les éléments. Ces informations sont tout à fait disponible sur Internet comme les descriptions de produits sur Amazon ou bien encore les tags sur les film dans IMDb (Internet Movie Database), les images postées sur Instagram, etc... 
Les succès de la recommandation par démarrage à froid pour un élément sont la plupart dû à l’utilisation d’un modèle de partage d’espace latent, en supposant qu’un élément doit conserver la même représentation dimensionnelle basse pour ses interactions et pour les informations qu’ils contiennent. Les algorithmes apprennent  grâce à la projection de ces deux dernières sur le même point dans l’espace latent. Dans le cas d’un élément de démarrage à froid sans interaction avec les utilisateurs, le vecteur latent correspondant peut être prédit à l’aide des informations de son contenu. Ainsi, les interactions entre les utilisateurs peuvent être calculées sur la base des vecteurs latents correspondants.

Bien que ces systèmes de partage d’espace latent obtiennent un succès impressionnant dans la recommandation d’éléments de démarrage à froid. Dans certains cas, on rencontre des difficultés lors de la conception des fonctions de projection, ce qui fait l’objet de la plupart des recherches actuelles et doivent parfois être spécifiques à un système [Van den Oord et al., 2013; Wang et Wang, 2014]. En plus de cela, certains types d’informations nécessitent des méthodes de projection non triviales, d’où l’utilisation des réseaux de neurones profonds DNN afin d’entraîner  l’espace latent avec les contenus de l’élément. On peut être amener à du sur-apprentissage en raison de la complexité des projections quand les données sont faibles.  

Au lieu de chercher un espace latent commun, l’article propose la méthode WCF pour traiter le problème. En théorie, nous allons utiliser la distance de Wasserstein pour mesurer la distance entre la préférence d’un utilisateur sur ses éléments d’interaction et ceux à démarrage à froid. La figure 1 ci-dessous illustre cela.

p: préférence d’un utilisateur sur ses éléments interagis.

q: sa référence pour ceux qui démarrent à froid.

![fig1](https://user-images.githubusercontent.com/39281095/84706689-b616fa80-af5d-11ea-8933-5ef21209bd49.png)

	figure 1

## La distance de Wasserstein

Cette distance calcule la divergence ente p et q, ce qui donne la similarité entre deux ensemble d’éléments: {v1, v2, v3} et {c1, c2}. On peut extraire les similitudes des éléments grâce aux informations qu’ils contiennent. Et Et ensuite, la préférence de l’utilisateur sur les éléments de démarrage à froid peut être résolue facilement en minimisant la distance de Wasserstein. Comme le CF, le WCF va collecté les informations sur les préférences de nombreux autres utilisateurs (collaborant)  et va supposé que la préférence des utilisateurs a approximativement un rang inférieur par rapport à la distance de Wasserstein. Empiriquement, WCF peut encore améliorer ses performances de prédiction sur les éléments de démarrage à froid.

## 2 - Arrière-plans

La méthode la plus utilisée pour effectuer des filtrages collaboratifs est de représenter à la fois les éléments et les utilisateurs par des vecteurs latents afin que les évaluations puissent être reconstruites à partir d’eux. Plusieurs instanciations ont été proposés ces dernières années, mais la factorisation matricielle (MF) [Mnih and Salakhutdinov, 2007; Koren et al., 2009] reste le plus populaire en raison de sa simplicité et de son efficacité. Elle a en effet fait ses preuves, notamment lors de très grandes recommandations de  films [Koren et al., 2003] et de produits [Linden et al., 2003]. Des études récentes étendent le cadre MF pour la recommandation de démarrage à froid de documents en incorporant les informations sur le contenu de ces derniers. Ces méthodes de recommandation de démarrage à froid de document utilisent en majorité un modèle de partagent d’espace latent. Par exemple, Saveki te al. [2014] et Barjasteh et al. [2016] propose d’utiliser MF comme fonction de projection pour les interactions et les contenus de document, [Van den Oord et al., 2013] et [Wang et al., 2014] proposent d’utiliser les CNN pour apprendre les vecteurs latents de la musique à partir de leurs signaux audios,…
	
La distance de Wasserstein, qui provient de la théorie du transport optimal [Rubner et al., 1998; Levina and Bickel, 2001], est une métrique de distance dans un espace probabiliste. Elle est capable de tirer parti des informations sur l’espace des fonctionnalités. Elle a été appliquée avec succès à de nombreuses applications, telles que la vision par ordinateur, le traitement du langage naturel, et attire de plus en plus d’attention dans le milieu universitaire. Cependant, aucun travail antérieur n’a appliqué la théorie du transport optimal au filtrage collaboratif, en particulier, le problème de démarrage à froid.

## 3 - Le problème proposé

### 3 .1 -  Définition du problème

![fig2](https://user-images.githubusercontent.com/39281095/84706783-df378b00-af5d-11ea-899d-89a890125e17.png)
	
	figure2

Le problème qui se pose est illustré par la figure2 ci-dessus. Supposons U, un ensemble de m utilisateurs et un ensemble de n dont les éléments sont appelés «les éléments interagis». La matrice d’interaction est noté R ∈ R n×m. R(vu) ≥ 0 est l’interaction entre l’utilisateur u et l’élément v, qui peut être un score d’évaluation, un nombre de clics ou de temps de visualisation, etc. R est habituellement une matrice d’observation et elle est parcimonieuse. Dans la partie droite, supposons C = {c1, c2, ..cs}, un ensemble d’éléments de démarrage à froid sans interaction avec tout les utilisateurs, et C∩V =  ∅.
Nous allons chercher une liste classée Lu = (cu1, cu2, …, cun) à partir C pour un utilisateur u, de telle sorte que  le rang de ‘cl’ soit supérieur  ‘cr’ dans Lu si l’utilisateur u préfère ‘cl’ à ‘cr’. 

### 3 .2 – La distance de Wasserstein pour la préférence des utilisateurs

#### Représentation des préférences de l’utilisateur

Soit ![](https://quicklatex.com/cache3/99/ql_08cc7df85964d4913b72a79d67a5e699_l3.png) qui désigne un simplex de dimension (n-1).
On va modéliser la préférence d'un utilisateur sur V comme une distribution de probabilité p ∈ Σn que V, où p(Vl) > p(Vr) si et seulement si l'utilisateur préfère Vl plus que Vr. De même pour la préférence de l'utilisateur sur C. En raison de la présence d'informations dans les éléments, on ne peut utiliser des mesures de distance telles : distance eulcidienne ou valeur cosinus de leur interaction. 

#### Distance de préférence utilisateur

La distance de Wasserstein réuni la similitude des éléments pour évaluer la distance de la préférence de l'utilisateur p sur V et q sur C. La distance de Wasserstein définit d'abord un polytope de plans de transport entre p et q comme suit:

![](https://quicklatex.com/cache3/e0/ql_5d2a296f403db3ba5a7e1fe06ac8c1e0_l3.png)

La distance de Wasserstein entre p et q est alors défini comme:

![](https://quicklatex.com/cache3/3b/ql_54cf4c1ea13f7ebf0729da3b6aae4c3b_l3.png)

La matrice de coût ici est ![](https://quicklatex.com/cache3/1e/ql_8148c05de5d5fef36566b3379f6f3b1e_l3.png), dont l'entré ![](https://quicklatex.com/cache3/2c/ql_52faa3d4f582aebd6b905c0dd1c9df2c_l3.png) évalue la différence entre les éléments ![](https://quicklatex.com/cache3/c2/ql_365c255d99078d4c7d39ff922a62ddc2_l3.png) et ![](https://quicklatex.com/cache3/06/ql_af642f1789ff5de0a525695b7d26f006_l3.png), T est le plan de transport.

##### Visualisation 
Pour pouvoir adapter la distance de Wasserstein à notre problème. On va redéfinir M comme étant "la matrice des coûts utilitaires" et T comme étant "le plan d'échange". En effet, en introduisant ce concept, dès que les intéractions d'un utilisateur correspondent à sa préférence, on pourra maximiser l'utilitaire de la personne.   
	
	Illustration 

![il1](https://user-images.githubusercontent.com/39281095/85161053-8159ba80-b25f-11ea-9fb4-efa11f246fe7.png)

![il2](https://user-images.githubusercontent.com/39281095/85161007-7868e900-b25f-11ea-9930-5ce839f24540.png)
