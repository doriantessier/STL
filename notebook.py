import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""#3D Geometry File Formats""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## About STL

        STL is a simple file format which describes 3D objects as a collection of triangles.
        The acronym STL stands for "Simple Triangle Language", "Standard Tesselation Language" or "STereoLitography"[^1].

        [^1]: STL was invented for – and is still widely used – for 3D printing.
        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(mo):
    with open("data/teapot.stl", mode="rt", encoding="utf-8") as _file:
        teapot_stl = _file.read()

    teapot_stl_excerpt = teapot_stl[:723] + "..." + teapot_stl[-366:]

    mo.md(
        f"""
    ## STL ASCII Format

    The `data/teapot.stl` file provides an example of the STL ASCII format. It is quite large (more than 60000 lines) and looks like that:
    """
    +
    f"""```
    {teapot_stl_excerpt}
    ```
    """
    +

    """
    """
    )
    return teapot_stl, teapot_stl_excerpt


@app.cell
def __(mo):
    mo.md(f"""

      - Study the [{mo.icon("mdi:wikipedia")} STL (file format)](https://en.wikipedia.org/wiki/STL_(file_format)) page (or other online references) to become familiar the format.

      - Create a STL ASCII file `"data/cube.stl"` that represents a cube of unit length  
        (💡 in the simplest version, you will need 12 different facets).

      - Display the result with the function `show` (make sure to check different angles).
    """)
    return


@app.cell
def __(show):
    file = open("data/cube.stl",'w')
    file.write("""solid cube
      facet normal 0 0 -1
        outer loop
          vertex 0 0 0
          vertex 1 0 0
          vertex 1 1 0
        endloop
      endfacet
      facet normal 0 0 -1
        outer loop
          vertex 0 0 0
          vertex 1 1 0
          vertex 0 1 0
        endloop
      endfacet
      facet normal 0 0 1
        outer loop
          vertex 0 0 1
          vertex 1 1 1
          vertex 1 0 1
        endloop
      endfacet
      facet normal 0 0 1
        outer loop
          vertex 0 0 1
          vertex 0 1 1
          vertex 1 1 1
        endloop
      endfacet
      facet normal -1 0 0
        outer loop
          vertex 0 0 0
          vertex 0 1 0
          vertex 0 1 1
        endloop
      endfacet
      facet normal -1 0 0
        outer loop
          vertex 0 0 0
          vertex 0 1 1
          vertex 0 0 1
        endloop
      endfacet
      facet normal 1 0 0
        outer loop
          vertex 1 0 0
          vertex 1 1 1
          vertex 1 1 0
        endloop
      endfacet
      facet normal 1 0 0
        outer loop
          vertex 1 0 0
          vertex 1 0 1
          vertex 1 1 1
        endloop
      endfacet
      facet normal 0 -1 0
        outer loop
          vertex 0 0 0
          vertex 0 0 1
          vertex 1 0 1
        endloop
      endfacet
      facet normal 0 -1 0
        outer loop
          vertex 0 0 0
          vertex 1 0 1
          vertex 1 0 0
        endloop
      endfacet
      facet normal 0 1 0
        outer loop
          vertex 0 1 0
          vertex 1 1 1
          vertex 0 1 1
        endloop
      endfacet
      facet normal 0 1 0
        outer loop
          vertex 0 1 0
          vertex 1 1 0
          vertex 1 1 1
        endloop
      endfacet
    endsolid cube""")

    file.close()
    show("data/cube.stl", theta = 0, phi = 0)
    return (file,)


@app.cell
def __(mo):
    mo.md(r"""## STL & NumPy""")
    return


@app.cell
def __(mo):
    mo.md(rf"""

    ### NumPy to STL

    Implement the following function:

    ```python
    def make_STL(triangles, normals=None, name=""):
        pass # 🚧 TODO!
    ```

    #### Parameters

      - `triangles` is a NumPy array of shape `(n, 3, 3)` and data type `np.float32`,
         which represents a sequence of `n` triangles (`triangles[i, j, k]` represents 
         is the `k`th coordinate of the `j`th point of the `i`th triangle)

      - `normals` is a NumPy array of shape `(n, 3)` and data type `np.float32`;
         `normals[i]` represents the outer unit normal to the `i`th facet.
         If `normals` is not specified, it should be computed from `triangles` using the 
         [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

      - `name` is the (optional) solid name embedded in the STL ASCII file.

    #### Returns

      - The STL ASCII description of the solid as a string.

    #### Example

    Given the two triangles that make up a flat square:

    ```python

    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ```

    then printing `make_STL(square_triangles, name="square")` yields
    ```
    solid square
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 0.0 0.0 0.0
          vertex 1.0 0.0 0.0
          vertex 0.0 1.0 0.0
        endloop
      endfacet
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 1.0 1.0 0.0
          vertex 0.0 1.0 0.0
          vertex 1.0 0.0 0.0
        endloop
      endfacet
    endsolid square
    ```

    """)
    return


@app.cell
def __(np):
    def normale(triangles):
            vect1 = triangles[:, 1] - triangles[:, 0]
            vect2 = triangles[:, 2] - triangles[:, 0]
            cross = np.cross(vect1, vect2)  #on fait le produit vectoriel pour avoir la normale
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            return (cross / norms)  #on normalise


    def make_STL(triangles, normals=None, name=""):
        if normals is None:
            normals = normale(triangles)

        stl_lines = [f"solid {name}"]
        for i, triangle in enumerate(triangles):
            normal = normals[i]
            stl_lines.append(f"  facet normal {normal[0]} {normal[1]} {normal[2]}")
            stl_lines.append("    outer loop")
            for vertex in triangle:
                stl_lines.append(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}")
            stl_lines.append("    endloop")
            stl_lines.append("  endfacet")
        stl_lines.append(f"endsolid {name}")

        return "\n".join(stl_lines)


    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    make_STL(square_triangles, name="square")
    return make_STL, normale, square_triangles


@app.cell
def __(mo):
    mo.md(
        """
        ### STL to NumPy

        Implement a `tokenize` function


        ```python
        def tokenize(stl):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `stl`: a Python string that represents a STL ASCII model.

        #### Returns

          - `tokens`: a list of STL keywords (`solid`, `facet`, etc.) and `np.float32` numbers.

        #### Example

        For the ASCII representation the square `data/square.stl`, printing the tokens with

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        print(tokens)
        ```

        yields

        ```python
        ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
        ```
        """
    )
    return


@app.cell
def __(np):
    def is_numeric(token):
        try:
            #tente de convertir le token en float
            float(token)
            return True
        except ValueError:
            return False

    def tokenize(stl):
        stl_array = np.array(list(stl))

        #on divise la chaine en morceaux selon les espaces (y compris les tabulations et les retours à la ligne)
        tokens1 = "".join(stl_array).split()

        #on fait un récap des mots utilisés en STL
        keywords = {"solid", "facet", "normal", "outer", "loop", 
                    "vertex", "endloop", "endfacet", "endsolid"}

        tokens = []
        solid_name = None
        for i, token in enumerate(tokens1):
            if token == "solid" and i + 1 < len(tokens1):
                #le token suivant "solid" est le nom du solide
                solid_name = tokens1[i+1]
                tokens.append(token) 
                tokens.append(solid_name) 
            elif token in keywords:
                tokens.append(token)
            elif is_numeric(token):
                tokens.append(np.float32(token))  #on convertit les nombres en float

        return tokens


    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
        square_stl = square_file.read()
    tokens = tokenize(square_stl)
    print(tokens)
    return is_numeric, square_file, square_stl, tokenize, tokens


@app.cell
def __(mo):
    mo.md(
        """
        Implement a `parse` function


        ```python
        def parse(tokens):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `tokens`: a list of tokens

        #### Returns

        A `triangles, normals, name` triple where

          - `triangles`: a `(n, 3, 3)` NumPy array with data type `np.float32`,

          - `normals`: a `(n, 3)` NumPy array with data type `np.float32`,

          - `name`: a Python string.

        #### Example

        For the ASCII representation `square_stl` of the square,
        tokenizing then parsing

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        triangles, normals, name = parse(tokens)
        print(repr(triangles))
        print(repr(normals))
        print(repr(name))
        ```

        yields

        ```python
        array([[[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]],

               [[1., 1., 0.],
                [0., 1., 0.],
                [1., 0., 0.]]], dtype=float32)
        array([[0., 0., 1.],
               [0., 0., 1.]], dtype=float32)
        'square'
        ```
        """
    )
    return


@app.cell
def __(np, tokenize):
    def parse(tokens):
        name = tokens[1]  #on ajoute le nom du solide qui est le 2ème élément du fichier
        triangles = []
        normals = []

        i = 2 #i=0 et i=1 ont déjà été traités

        while i < len(tokens):
            if tokens[i] == "facet" and tokens[i+1] == "normal":
                #on ajoute le vecteur normal
                normal = [tokens[i+2], tokens[i+3], tokens[i+4]]
                normals.append(normal)
                i += 5  #on saute les éléments déjà traités

                i += 2  #on saute les éléments "outer" et "loop"

                #maintenant on s'occupe des coordonnées du triangle
                sommets = []
                for _ in range(3):
                    coords = [tokens[i+1], tokens[i+2], tokens[i+3]]
                    sommets.append(coords)
                    i += 4  #on saute les éléments déjà traités

                triangles.append(sommets)

                i+=2 #on saute les termes "endloop" et "endfacet"

            elif tokens[i] == "endsolid" :
                i+=1
                break
            else:
                raise ValueError(f"Unexpected token: {i, tokens[i]}")

        #on convertit en nd.array
        triangles = np.array(triangles, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)

        return triangles, normals, name


    with open("data/square.stl", mode="rt", encoding="us-ascii") as _square_file:
        _square_stl = _square_file.read()
    _tokens = tokenize(_square_stl)
    triangles, normals, name = parse(_tokens)
    print(repr(triangles))
    print(repr(normals))
    print(repr(name))
    return name, normals, parse, triangles


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Rules & Diagnostics



        Make diagnostic functions that check whether a STL model satisfies the following rules

          - **Positive octant rule.** All vertex coordinates are non-negative.

          - **Orientation rule.** All normals are (approximately) unit vectors and follow the [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

          - **Shared edge rule.** Each triangle edge appears exactly twice.

          - **Ascending rule.** the z-coordinates of (the barycenter of) each triangle are a non-decreasing sequence.

    When the rule is broken, make sure to display some sensible quantitative measure of the violation (in %).

    For the record, the `data/teapot.STL` file:

      - 🔴 does not obey the positive octant rule,
      - 🟠 almost obeys the orientation rule, 
      - 🟢 obeys the shared edge rule,
      - 🔴 does not obey the ascending rule.

    Check that your `data/cube.stl` file does follow all these rules, or modify it accordingly!

    """
    )
    return


@app.cell
def __(np):
    def positive_octant_rule(triangles):
        negatif = np.sum(triangles < 0)
        coord_tot = triangles.size
        pourcentage = 100 * negatif / coord_tot
        return negatif == 0, pourcentage


    def orientation_rule(triangles, normals):
        #on vérifie d'abord si les vecteurs sont unitaires, à 10-8 près
        norme = np.linalg.norm(normals, axis=1)
        norme_mauvaise = np.sum(np.abs(norme - 1) > 1e-8)

        #on regarde la règle de la main droite
        cote1 = triangles[:, 1] - triangles[:, 0]
        cote2 = triangles[:, 2] - triangles[:, 0]
        normale = np.cross(cote1, cote2)
        normale = normale / np.linalg.norm(normale, axis=1, keepdims=True)
        non_main_droite = np.sum(np.dot(normals, normale.T).diagonal() < 0)

        total_normale = normals.shape[0]
        pourcentage = 100 * (norme_mauvaise + non_main_droite) / total_normale
        return norme_mauvaise == 0 and non_main_droite == 0, pourcentage


    def shared_edge_rule(triangles):
        #on crée les côtés
        cotes = np.concatenate([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]]
        ])

        cotes = np.sort(cotes, axis=1)

        #on relève les cotés n'apparaissant pas strictement 2 fois
        _, counts = np.unique(cotes, axis=0, return_counts=True)
        erreurs = np.sum(counts != 2)
        cotes_tot = len(cotes)
        pourcentage = 100 * erreurs / cotes_tot
        return erreurs == 0, pourcentage


    def ascending_rule(triangles):
        #on calcule la coordonnée selon z du barycentre, puis on vérifie qu'ils ne décroissent pas
        barycentre_z = np.mean(triangles[:, :, 2], axis=1)

        erreurs = np.sum(np.diff(barycentre_z) < 0)
        triangles_tot = len(barycentre_z)
        pourcentage = 100 * erreurs / (triangles_tot - 1) if triangles_tot > 1 else 0
        return erreurs == 0, pourcentage


    def bilan(triangles, normals):
        results = {}
        results["Positive Octant Rule"] = positive_octant_rule(triangles)
        results["Orientation Rule"] = orientation_rule(triangles, normals)
        results["Shared Edge Rule"] = shared_edge_rule(triangles)
        results["Ascending Rule"] = ascending_rule(triangles)
        return results
    return (
        ascending_rule,
        bilan,
        orientation_rule,
        positive_octant_rule,
        shared_edge_rule,
    )


@app.cell
def __(bilan, parse, tokenize):
    with open("data/cube.stl", mode="rt", encoding="us-ascii") as _cube_file:
        _cube_stl = _cube_file.read()
    _tokens = tokenize(_cube_stl)
    triangles1, normals1, name1 = parse(_tokens)

    print(bilan(triangles1, normals1))

    with open("data/teapot.stl", mode="rt", encoding="us-ascii") as _teapot_file:
        _teapot_stl = _teapot_file.read()
    _tokens = tokenize(_teapot_stl)
    triangles2, normals2, name2 = parse(_tokens)

    print(bilan(triangles2, normals2))
    return name1, name2, normals1, normals2, triangles1, triangles2


@app.cell
def __(bilan, normals1, np, triangles1):
    #il faut modifier les fichier cube

    def fix_cube(triangles, normals):
        #règle 2
        cote = triangles[:, 1:] - triangles[:, :1]  #on prend 2 côtés pour le prouit vectoriel
        nouvelles_norm = np.cross(cote[:, 0], cote[:, 1])
        nouvelles_norm /= np.linalg.norm(nouvelles_norm, axis=1, keepdims=True)
        normals = nouvelles_norm

        #la règle 3 est résolue par la même occasion
        #règle 4
        barycentre_z = np.mean(triangles[:, :, 2], axis=1)
        indices_tris = np.argsort(barycentre_z)
        triangles = triangles[indices_tris]
        normals = normals[indices_tris]

        return triangles, normals


    #on applique à notre fonction au cube
    bons_triangles, bonnes_normales = fix_cube(triangles1, normals1)

    #on refait le bilan
    bon_diagnostique = bilan(bons_triangles, bonnes_normales)

    #on montre les résultats
    bons_triangles, bonnes_normales, bon_diagnostique
    return bon_diagnostique, bonnes_normales, bons_triangles, fix_cube


@app.cell
def __(mo):
    mo.md(
    rf"""
    ## OBJ Format

    The OBJ format is an alternative to the STL format that looks like this:

    ```
    # OBJ file format with ext .obj
    # vertex count = 2503
    # face count = 4968
    v -3.4101800e-003 1.3031957e-001 2.1754370e-002
    v -8.1719160e-002 1.5250145e-001 2.9656090e-002
    v -3.0543480e-002 1.2477885e-001 1.0983400e-003
    v -2.4901590e-002 1.1211138e-001 3.7560240e-002
    v -1.8405680e-002 1.7843055e-001 -2.4219580e-002
    ...
    f 2187 2188 2194
    f 2308 2315 2300
    f 2407 2375 2362
    f 2443 2420 2503
    f 2420 2411 2503
    ```

    This content is an excerpt from the `data/bunny.obj` file.

    """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.obj", scale="1.5"))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Study the specification of the OBJ format (search for suitable sources online),
        then develop a `OBJ_to_STL` function that is rich enough to convert the OBJ bunny file into a STL bunny file.
        """
    )
    return


@app.cell
def __(make_STL, np):
    def obj_to_stl(OBJ, STL):

        sommets = []
        faces = []

        #on lit le fichier OBJ et on extrait les sommets et les normales
        with open(OBJ, 'r') as obj:
            for line in obj:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    #c'est un sommet
                    sommets.append([float(coord) for coord in parts[1:4]])
                elif parts[0] == 'f':
                    #c'est une face (on met -1 car en OBJ le compte commence à 1, pas à 0)
                    face = [int(x.split('/')[0]) - 1 for x in parts[1:4]]
                    faces.append(face)

        sommets = np.array(sommets, dtype=np.float32)
        faces = np.array(faces, dtype=int)

        #On calcule les normales par produit vectoriel, puis on normalise
        triangles = sommets[faces]
        vect1 = triangles[:, 1] - triangles[:, 0]
        vect2 = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(vect1, vect2)
        normals = normals/np.linalg.norm(normals, axis=1, keepdims=True)

        #On écrit en STL
        stl = make_STL(triangles, normals, STL)
        with open(f"{STL}.stl", "w", encoding="ascii") as file:
            file.write(stl)


        return stl
    return (obj_to_stl,)


@app.cell
def __(obj_to_stl):
    obj_to_stl("data/bunny.obj", "bunny")
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Binary STL

    Since the STL ASCII format can lead to very large files when there is a large number of facets, there is an alternate, binary version of the STL format which is more compact.

    Read about this variant online, then implement the function

    ```python
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        pass  # 🚧 TODO!
    ```

    that will convert a binary STL file to a ASCII STL file. Make sure that your function works with the binary `data/dragon.stl` file which is an example of STL binary format.

    💡 The `np.fromfile` function may come in handy.

        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        stl_text = make_STL(faces, normals)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)
    return (STL_binary_to_text,)


@app.cell
def __(STL_binary_to_text):
    STL_binary_to_text("data/dragon.stl","dragon.txt")
    return


@app.cell
def __(mo):
    mo.md(rf"""## Constructive Solid Geometry (CSG)

    Have a look at the documentation of [{mo.icon("mdi:github")}fogleman/sdf](https://github.com/fogleman/) and study the basics. At the very least, make sure that you understand what the code below does:
    """)
    return


@app.cell
def __(X, Y, Z, box, cylinder, mo, show, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    mo.show_code(show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg,)


@app.cell
def __(mo):
    mo.md("""ℹ️ **Remark.** The same result can be achieved in a more procedural style, with:""")
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    mo,
    orient,
    show,
    sphere,
    union,
):
    demo_csg_alt = difference(
        intersection(
            sphere(1),
            box(1.5),
        ),
        union(
            orient(cylinder(0.5), [1.0, 0.0, 0.0]),
            orient(cylinder(0.5), [0.0, 1.0, 0.0]),
            orient(cylinder(0.5), [0.0, 0.0, 1.0]),
        ),
    )
    demo_csg_alt.save("output/demo-csg-alt.stl", step=0.05)
    mo.show_code(show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg_alt,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## JupyterCAD

    [JupyterCAD](https://github.com/jupytercad/JupyterCAD) is an extension of the Jupyter lab for 3D geometry modeling.

      - Use it to create a JCAD model that correspond closely to the `output/demo_csg` model;
    save it as `data/demo_jcad.jcad`.

      - Study the format used to represent JupyterCAD files (💡 you can explore the contents of the previous file, but you may need to create some simpler models to begin with).

      - When you are ready, create a `jcad_to_stl` function that understand enough of the JupyterCAD format to convert `"data/demo_jcad.jcad"` into some corresponding STL file.
    (💡 do not tesselate the JupyterCAD model by yourself, instead use the `sdf` library!)


        """
    )
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    json,
    orient,
    sphere,
    union,
):
    def jcad_to_sdf(jcad_file):
        with open(jcad_file, 'r') as file:
            jcad_data = json.load(file)

        objets = {}

        #on crée les formes
        def create_sphere(params):
            radius = params.get("Radius", 1)
            return sphere(radius)

        def create_box(params):
            size = [
                params.get("Length", 1),
                params.get("Width", 1),
                params.get("Height", 1)
            ]
            return box(size)

        def create_cylinder(params):
            radius = params.get("Radius", 1)
            orientation = params.get("Placement", {}).get("Axis", 1)
            return orient(cylinder(radius), orientation)

        #dico des formes existantes
        formes = {
            "Part::Sphere": create_sphere,
            "Part::Box": create_box,
            "Part::Cylinder": create_cylinder
        }

        #on construit les objets
        for obj in jcad_data["objects"]:
            forme = obj["shape"]
            params = obj.get("parameters", {})
            nom = obj["name"]

            if forme in formes:
                objets[nom] = formes[forme](params)

            elif forme == "Part::MultiCommon":
                #intersection
                dependencies = obj.get("dependencies", [])
                objets[nom] = intersection(*[objets[dep] for dep in dependencies]) 
                #* permet de spérarer les valeurs de la liste

            elif forme == "Part::Cut":
                #différence
                base = objets[obj["parameters"]["Base"]]
                tool = objets[obj["parameters"]["Tool"]]
                objets[nom] = difference(base,tool)

            elif forme == "Part::MultiFuse":
                #union
                dependencies = obj.get("dependencies", [])
                objets[nom] = union(*[objets[dep] for dep in dependencies])

        #on renvoie la forme finale (que j'ai appelé exprès "Final form")
        return objets["Final form"]

    def jcad_to_stl(jcad_file, nom_sortie):
        #d'abord on convertit en sdf...
        fichier_sdf = jcad_to_sdf(jcad_file)

        #...puis en stl  (comme dans la cellule précédente)
        fichier_sdf.save(nom_sortie)


    jcad_to_stl("data/demo_jcad.jcad", "data/demo_jcad.stl")
    return jcad_to_sdf, jcad_to_stl


@app.cell
def __(mo, show):
    mo.show_code(show("data/demo_jcad.stl", theta=45.0, phi=30.0))
    return


@app.cell
def __(mo):
    mo.md("""## Appendix""")
    return


@app.cell
def __(mo):
    mo.md("""### Dependencies""")
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera


    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference

    mo.show_code()
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


@app.cell
def __(mo):
    mo.md(r"""### STL Viewer""")
    return


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)

    mo.show_code()
    return (show,)


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


if __name__ == "__main__":
    app.run()
