# Yocto/Raytrace: Tiny Raytracer

In questo homework, impareremo i principi di base della sintesi di immagini
implementando un semplice path tracer. In particolare, impareremo come

- usare funzioni di intersezione di raggi,
- scrivere semplici shaders,
- scrivere un semplice renderer

## Framework

Il codice utilizza la libreria [**Yocto/GL**](https://github.com/xelatihy/yocto-gl),
che è inclusa in questo progetto nella directory `yocto`.
Si consiglia di consultare la documentazione della libreria.
Inoltre, dato che la libreria verrà migliorata durante il corso,
consigliamo di mettere **star e watch su github** in modo che
arrivino le notifiche di updates.

Il codice è compilabile attraverso [Xcode](https://apps.apple.com/it/app/xcode/id497799835?mt=12)
su OsX e [Visual Studio 2019](https://visualstudio.microsoft.com/it/vs/) su Windows,
con i tools [cmake](www.cmake.org) e [ninja](https://ninja-build.org)
come mostrato in classe e riassunto, per OsX,
nello script due scripts `scripts/build.sh`.
Per compilare il codice è necessario installare Xcode su OsX e
Visual Studio 2019 per Windows, ed anche i tools cmake e ninja.
Come discusso in classe, si consiglia l'utilizzo di
[Visual Studio Code](https://code.visualstudio.com), con i plugins
[C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) e
[CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
extensions, che è già predisposto per lavorare su questo progetto.

Questo homework consiste nello sviluppare vari shaders che, per semplicità,
sono tutti contenuti nel file `yocto_raytrace.cpp`. Gli shader che scriverete
sono eseguiti per ogni pixel dell'immagine dalla funzione `render_samples()`
che diamo già implementata. A sua volta questa funzione è chiamata da
`yraytrace.cpp` come interfaccia a riga di comando, e da `yiraytraces.cpp`
che mostra una semplice interfaccia grafica.

Questo repository contiene anche tests che sono eseguibili da riga di comando
come mostrato in `run.sh`. Le immagini generate dal runner sono depositate
nella directory `out/`. Questi risultati devono combaciare con le immagini nella
directory `check/`. Per la consegna dell'homework basta renderizzare solo le
immagini a bassa risoluzione che andranno nella directory `out/lowres`.
In `check/highres` ci sono immagini calcolate ad alta risoluzione per farvi
vedere il miglioramento.

## Funzionalità (24 punti)

In this homework you will implement the following features:

- **Color Shader** nella funzione `shade_color()`:
  - implementare uno shader che ritorna il colore del materiale del punto di intersezione
  - use `intersect_scene_bvh()` for intersection
- **Normal Shader** nella funzione `shade_normal()`:
  - implementare uno shader che ritorna la normale del punto di intersezione,
    tradotta in colore aggiungendo 0.5 e moltiplicando per 0.5
- **Texcoord Shader** nella funzione `shade_texcoord()`:
  - implementare uno shader che ritorna le texture coordinates del punto di intersezione,
    tradotte in colori per i canali RG; usare la funzione `fmod()` per forzarle
    nel range [0, 1]
- **Eyelight Shader** nella funzione `shade_eyelight()`:
  - implementare uno shader che calcola il diffuse shading assumendo di avere
    una fonte di illuminazione nelle fotocamera
- **Raytrace Shader** nella funzione `shade_raytrace()`:
  - implementare uno shader ricorsivo che simula l'illuminazione per una varietà
    di materiali seguendo gli steps descritti nelle lecture notes riassunti qui
    di seguito
  - calcolare posizione, normale and texture coordinates
  - calcolare il valore dei materiali multiplicando le constanti trovate
    nel material con i valori nelle textures
  - implementare i seguenti materiali nell'ordinane presentato nelle slides:
    polished transmission, polished metals, rough metals, rough plastic, and matte
  - potete usare tutte le funzioni di Yocto/Shading ed in particolare `fresnel_schlick()`,
    e tutte le `sample_XXX()`
  - per le supeerfici matte, usa la funzione `sample_hemisphere_cos()` e ignora
    la normalizzazione di 2 pi
  - per i metalli rough, implementeremo una versione semplificata che utilizza
    una microfacet normal scelta in modo accurato, ma poi si comparta come se
    fosse un metallo polished che usa questa normale
    - `mnormal = sample_hemisphere_cospower(exponent, normal, rand2f(rng));`
    - `exponent   = 2 / (roughness * roughness)`
  - per le superifici plastiche, utilizziamo la stessa normale e poi un test con
    Fresnel che sceglie se fare la riflessione o la diffusa

## Extra Credit (10 punti)

- **refraction** (facile): implementaree la rifrazione polished nella funzione
  `shade_raytrace()`:
  - implementare la rifrazione usando `refract()` per la direzione,
  - `ior` per ottenere l'indice di rifrazione dalla riflettività (0.04)
  - e ricordandosi di invertire l'indice di rifrazione quando si lascia la supereficie
  - potete inspifrarvi a [Raytracing in one Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics)
- **cell shading** (medio): implementare uno shader non realistico seguendo
  [cell shading](https://roystan.net/articles/toon-shader.html)
- **matcap shading** (medio): implemntare uno shader non realistico seguendo
  [matcap shading 1](http://viclw17.github.io/2016/05/01/MatCap-Shader-Showcase/)
  o [matcap shading 2](https://github.com/hughsk/matcap)
- **your own shader** (medio): scrivete altri shaders basati su idee
  trovate su [ShaderToy](www.shadertoy.com)
- **other BVHs** (medio): il tempo di esecuzione è dominato da ray-scene
  intersection; testiamo altri BVH per vedere se ci sono speed up significativi
  - abbiamo già integrato Intel Embree, quindi questo non conta
  - integrare il BVh [nanort](https://github.com/lighttransport/nanort) o
    [bvh](https://github.com/madmann91/bvh)
  - testare le scene più larghe e riportare nel readme i tempi confrontandoli
    con i nostri
  - per integrarli, potete scrivere nuovi shaders o inserire i nuovi Bvh
    dentro il nostro bvh
- **thick lines and points** (difficile): il rendering di punti è linee è
  attulamente approssimato; in questo homework andremo a calcolarli in modo corretto
  - renderizzare punti come sfere e linee come capped cones
  - algoritmi di intersezione si trovano [qui](https://iquilezles.org/www/articles/intersectors/intersectors.htm) con i nomi Sphere e Rounded Cone
  - per integrarli nel nostro codice si deve (a) inserire le funzioni di 
    intersezione `intersect_bvh` in `yocto_bvh`, (b) cambiare le funzion
    `eval_position` e `eval_normal` con le normali appropriate, oppure (c)
    cambiare i dati che ritorna l'intersezione e aggiungere normal e position
  - se si implementa questo punto, chiedere al docente file di tests 
- **volume rendering** (difficile): implementare il rendering di volumi semplici seguendo
    [Ray tracing The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html#volumes)
- **nuove scene** (molto facile, pochi punti): creare scene aggiuntive
  - creare immagini aggiuntive con il vostro renderer da modelli assemblati da voi
  - per creare modelli nuovi, potete editare scene in Blender e salvarle in glTF,
    usando `yscene` di Yocto per poi aggiustare a mano se serve
  - il vostro renderer fa molto bene environment maps e materiali diffusi
    quindi io mi concentrerei su quelli - trovate ottime environment maps su
    [PolyHaven](https://polyhaven.com)
  - come punto di partenza potete assemblare nuove scene mettendo nuovi
    modelli 3D, textures e environment maps - as esempio da
    [PolyHaven](https://polyhaven.com) o [Sketchfab](https://sketchfab.com)
  
Per la consegna, il contenuto dell'extra credit va descritto in un PDF chiamato
**readme.pdf**, in aggiunta alla consegna di codice e immagini.
Basta mettere per ogni extra credit, il nome dell'extra credit,
un paragrafo che descriva cosa avete implementato, una o più immagini dei
risulato, e links a risorse utilizzate nell'implementazione.

Potete produrre il PDF con qualsiasi strumento. Una possibilità comoda è di
scrivere il file in Markdown e convertilo in PDF con VSCode plugins o altri
strumenti che usate online. In questo modo potete linkare direttamente alla
immagini che consegnate.

## Istruzioni

Per consegnare l'homework è necessario inviare una ZIP che include il codice e
le immagini generate, cioè uno zip _con le sole directories `yocto_raytrace` e `out`_.
Per chi fa l'extra credit, includere anche `apps`, ulteriori immagini, e un
file **`readme.pdf`** che descrive la lista di extra credit implementati.
Per chi crea nuovi modelli, includere anche le directory con i nuovo modelli creati.
Il file va chiamato `<cognome>_<nome>_<numero_di_matricola>.zip`
e vanno escluse tutte le altre directory. Inviare il file su Google Classroom.
