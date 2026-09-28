import bpy
import numpy as np
import bmesh
import os
import sys
import inspect
import time
import json
import math
import pathlib
from ast import literal_eval

import subprocess
import site


#import open3d as o3d


LONG = True
LONG = False
if LONG:
    D_TYPE_F = np.float64
    D_TYPE_I = np.int64
else:
    D_TYPE_F = np.float32
    #D_TYPE_F = np.float64
    D_TYPE_I = np.int32


# Blender Kit Creds:
# kr@origin4n6.com
# @!jT8jQDR@D6BXhj



#=======================#
# MATH ---------------#
#=======================#
# math
def curve_gen(scalars, type=0, height=1):
    """Takes points between zero and 1 and plots them on a curve"""
    if type == 0: # smooth middle
        mid = scalars ** 2
        mid_flip = (-scalars + 1) ** 2
        return mid * mid_flip * 16 * height
        
    if type == 1: # half circle
        return np.sqrt(scalars) * np.sqrt(-scalars + 1)
    
    if type == 2: # smooth bottom to top
        reverse = -scalars + 1
        c1 = (scalars ** 2) * reverse
        c2 = (-reverse ** 2 + 1) * scalars
        smooth = c1 + c2
        return smooth

    if type == 3: # smooth top to bottom flip
        reverse = -scalars + 1
        c1 = (scalars ** 2) * reverse
        c2 = (-reverse ** 2 + 1) * scalars
        smooth = c1 + c2
        return -smooth + 1

    if type == 4: # 1/4 circle top left
        return np.sqrt(-scalars + 2) * np.sqrt(scalars)

    if type == 5: #1/4 circle bottom right
        x = np.sqrt(-scalars + 1) * np.sqrt(scalars + 1)
        return -x + 1

    if type == 6: #1/4 circle bottom left
        return -(np.sqrt(-scalars + 2) * np.sqrt(scalars)) + 1

    if type == 7: #1/4 circle top right
        x = np.sqrt(-scalars + 1) * np.sqrt(scalars + 1)
        return x 


# math
def map_ranges(r1b, r1t, r2b, r2t, val):
    """Find the value of range 1 where
    it maps to range two.
    r1b: range 1 bottom
    r1t: range 1 top"""

    dif1 = r1t - r1b
    dif2 = r2t - r2b
    vd = val - r1b
    dv1 = vd / dif1
    
    return dif2 * dv1 + r2b


#=======================#
# PYTHON ---------------#
#=======================#


def benchmark_to_text(text_name: str,
                      line_number: int,
                      func,
                      iterations: int = 1000,
                      label: str = None,
                      compare_total: float = None,
                      compare_label: str = None):
    if label is None:
        label = func.__name__
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / iterations
    message = ("# " + f"{label} | Iterations: {iterations} | "
               f"Total: {total_time:.6f}s | "
               f"Avg: {avg_time:.8f}s")
    write_to_text(text_name, line_number - 1, message)

    if compare_total is not None:
        compare_avg = compare_total / iterations
        if avg_time <= compare_avg:
            ratio = compare_avg / avg_time
            comparison = f"# {label} is {ratio:.2f}x faster than {compare_label or 'func2'}"
        else:
            ratio = avg_time / compare_avg
            comparison = f"# {compare_label or 'func2'} is {ratio:.2f}x faster than {label}"
        write_to_text(text_name, line_number, comparison)

    return total_time


def write_to_text(text_name: str, line_number: int, message: str):
    """
    Writes `message` to `line_number` (0-based) in the Blender
    text datablock `text_name`, without depending on cursor position.
    """

    text = bpy.data.texts.get(text_name)

    if text is None:
        text = bpy.data.texts.new(text_name)

    # Get full content as string
    content = text.as_string()

    # Split into lines safely
    lines = content.split("\n")

    # Ensure enough lines exist
    while len(lines) <= line_number:
        lines.append("")

    # Replace target line
    lines[line_number] = message

    # Rebuild entire text block
    text.from_string("\n".join(lines))
    text.current_character = text.current_character


# PIP STUFF ===================
    
# Packages whose dependencies Blender already ships.  pip --target does not
# look at what is importable, so without --no-deps installing scipy also drops
# a second numpy (60-odd MB) into the user's modules folder.  Blender's own
# numpy still wins on sys.path, so that copy is dead weight at best and a
# version clash waiting to happen at worst.
PIP_NO_DEPS = {"scipy"}          # needs numpy, which Blender bundles


def _wheel_files(path):
    try:
        return {f for f in os.listdir(path) if f.endswith(".whl")}
    except OSError:
        return set()


def external_lib(module="scipy", pip_spec=None, timeout=600):
    """pip-install `module` into Blender's user modules folder.

    Returns True if the module is importable afterwards.  Never raises: pip
    fails for all sorts of reasons outside our control (no network, a proxy, a
    locked-down machine) and none of them should take the addon down.
    """
    import importlib.util

    # Get a writable modules path for user-installed packages
    modules_path = bpy.utils.user_resource('SCRIPTS', path='modules', create=True)

    # Add the modules path to sys.path if not already there
    if modules_path not in sys.path:
        sys.path.append(modules_path)

    # Function to ensure pip is installed
    def ensure_pip():
        # Nothing in here may raise: this runs from module import time, and an
        # exception would take the whole addon down on a machine that simply
        # has no network.  Timeouts matter too -- pip against an unreachable
        # index will sit there forever and freeze Blender with no window.
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', '--version'],
                                  timeout=60)
            print("Pip is already installed.")
            return True
        except Exception:
            print("Bootstrapping pip...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'ensurepip', '--default-pip'],
                timeout=180)
            print("Pip installed successfully.")
            return True
        except Exception as e:
            print("Could not bootstrap pip: %s" % e)
            return False

    # Function to install a package if missing
    def install_if_missing(package_name, pip_spec):
        if have_module(package_name):
            print(f"{package_name} is already installed.")
            return True
        print(f"Installing {package_name}...")
        cmd = [sys.executable, '-m', 'pip', 'install', '--target', modules_path]
        if package_name in PIP_NO_DEPS:
            cmd.append('--no-deps')      # don't shadow Blender's own numpy
        cmd.append(pip_spec)
        before = _wheel_files(modules_path)
        try:
            subprocess.check_call(cmd, timeout=timeout)
        except Exception as e:
            # CalledProcessError, TimeoutExpired, FileNotFoundError,
            # PermissionError -- all of them mean the same thing to us
            print(f"Failed to install {package_name}. Error: {e}")
            return False

        # pip leaves the downloaded .whl sitting in --target (40-odd MB for
        # scipy); it is of no use once unpacked
        for f in _wheel_files(modules_path) - before:
            try:
                os.remove(os.path.join(modules_path, f))
            except OSError:
                pass

        # pip said yes; make sure python agrees before anyone relies on it
        importlib.invalidate_caches()
        if not have_module(package_name):
            print(f"{package_name} installed but is not importable from "
                  f"{modules_path}.")
            return False
        print(f"{package_name} installed successfully.")
        return True

    # have_module rather than find_spec: find_spec raises for a package whose
    # parent is broken or blocked, and nothing here may raise
    if have_module(module):
        return True
    if not ensure_pip():
        return False

    return install_if_missing(module, pip_spec or module)




def setup_pip_in_blender():
    """
    Sets up pip in Blender's Python environment and returns the pip executable path.
    """
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = os.path.join(sys.prefix, 'bin', 'python')

    # Get pip
    try:
        import pip
        return python_exe
    except ImportError:
        print("Pip not found. Installing pip...")
        
    # Download get-pip.py
    import urllib.request
    url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = os.path.join(os.path.dirname(__file__), "get-pip.py")
    
    urllib.request.urlretrieve(url, get_pip_path)
    
    # Install pip
    subprocess.call([python_exe, get_pip_path])
    
    # Clean up
    os.remove(get_pip_path)
    
    return python_exe


def install_package(package_name):
    external_lib(module=package_name)
    return have_module(package_name)


# ===== PANELS ===== #
# The sidebar tabs, named in one place because six modules draw into them.
MC_TAB = "MC5"                  # the simulation
MC_TOOLS_TAB = "MC5 Tools"      # building the mesh / pattern

# Panel order within a tab.  Blender sorts by bl_order then registration
# order, so stating it keeps the arrangement from depending on which module
# happened to register first.
ORDER = {"main": 0, "forces": 10, "collision": 20, "magnetic": 30,
         "wind": 40, "hooks": 50, "cache": 60, "presets": 70,
         "objects": 80, "settings": 90}


def active_mesh(context):
    """The active object when it is a mesh, else None.  Panels poll on this
    rather than drawing a "no object selected" placeholder."""
    ob = context.object
    return ob if (ob is not None and ob.type == 'MESH') else None


def active_cloth(context):
    """The active object when it is a mesh with cloth switched on, else None."""
    ob = active_mesh(context)
    if ob is None:
        return None
    try:
        return ob if ob.MC_props.cloth else None
    except AttributeError:
        return None             # properties not registered yet


def panel_note(layout, text, icon='INFO'):
    """A quiet line where a panel's controls would be.

    Panels stay in the sidebar whatever is selected -- both tabs and every
    panel are always there to be found -- so each one says what it is waiting
    for instead of vanishing."""
    col = layout.column()
    col.enabled = False
    col.scale_y = 0.8
    col.label(text=text, icon=icon)


def needs_mesh(layout, context):
    """Draw the placeholder if there is no mesh to work on.
    True means the caller has nothing to draw."""
    if active_mesh(context) is None:
        panel_note(layout, "Select a mesh object")
        return True
    return False


def needs_cloth(layout, context):
    """Same, for a panel whose settings belong to a cloth."""
    if active_mesh(context) is None:
        panel_note(layout, "Select a mesh object")
        return True
    if active_cloth(context) is None:
        panel_note(layout, "Turn on Cloth in the Main panel")
        return True
    return False


# ===== native library discovery ===== #
# The solver DLL has to be found in two quite different situations.  During
# development the modules are text datablocks inside a .blend and there is no
# __file__ at all; installed as an addon they are a real package in a folder
# Blender chose when it unzipped it.  Nothing may be hard-coded to one machine.
def lib_suffix(platform=None):
    """What a shared library is called on this platform."""
    p = platform or sys.platform
    if p.startswith("win"):
        return ".dll"
    if p == "darwin":
        return ".dylib"
    return ".so"


LIB_SUFFIX = lib_suffix()
# Named per platform rather than hard-coded to .dll, so a mac or linux build of
# the solver is found by the same search.  Only Windows binaries exist today.
DLL_STEMS = ("mc_cloth_solver", "cloth_solver")
DLL_NAMES = tuple(stem + LIB_SUFFIX for stem in DLL_STEMS)
DLL_SUBDIRS = ("", "lib", "bin", "dll_files")
# A folder of your own, searched first: for a library kept outside the addon,
# which is what the dll_path setting in the MC5 panel also does.  This replaced
# a developer's own path that was hard-coded here.
DLL_EXTRA = os.environ.get("MC_LIB_DIR", "")


def _dll_roots():
    """Every directory worth looking in, best first."""
    roots = []

    # 1. an explicit override, for developers and for odd installs
    try:
        p = bpy.context.scene.MC_props.dll_path
        if p:
            p = bpy.path.abspath(p)
            roots.append(p if os.path.isdir(p) else os.path.dirname(p))
    except Exception:
        pass

    # 2. next to this module when it is a real file (the addon case)
    try:
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass                                   # running as a text datablock

    # 3. where the text datablocks were loaded from, if they are linked to disk
    try:
        for t in bpy.data.texts:
            if t.filepath:
                d = os.path.dirname(bpy.path.abspath(t.filepath))
                if d:
                    roots.append(d)
    except Exception:
        pass

    # 4. beside the .blend, which is handy while developing
    try:
        if bpy.data.filepath:
            roots.append(os.path.dirname(bpy.data.filepath))
    except Exception:
        pass

    roots.append(DLL_EXTRA)

    # 5. In a checkout the libraries are built per platform into
    # addon/lib/<platform>/ , which is a sibling of the Python folder the
    # modules were loaded from.  Each platform folder is added, and only the one
    # holding files named for this platform's suffix can match anyway.
    for r in list(roots):
        roots += _lib_dirs_under(os.path.join(os.path.dirname(r), "addon", "lib"))
        roots += _lib_dirs_under(os.path.join(r, "addon", "lib"))
        roots += _lib_dirs_under(r)

    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _lib_dirs_under(path):
    """`path` and the directories directly inside it, when it exists."""
    if not path or not os.path.isdir(path):
        return []
    out = [path]
    try:
        out += [os.path.join(path, d) for d in sorted(os.listdir(path))
                if os.path.isdir(os.path.join(path, d))]
    except OSError:
        pass
    return out


def find_dll(names=DLL_NAMES, required=True):
    """Absolute path to the solver DLL, or None.

    Searched in order: an explicit path property, the addon folder, wherever
    the text datablocks came from, then the old hard-coded dev folder.
    """
    tried = []
    for root in _dll_roots():
        for sub in DLL_SUBDIRS:
            d = os.path.join(root, sub) if sub else root
            for name in names:
                p = os.path.join(d, name)
                tried.append(p)
                if os.path.isfile(p):
                    return p
    if required:
        print("MC: could not find the solver DLL. Looked in:")
        for p in tried[:12]:
            print("      %s" % p)
        if len(tried) > 12:
            print("      ...and %d more" % (len(tried) - 12))
    return None


# ===== external dependencies ===== #
# Blender ships numpy and nothing else we use, so scipy (and anything else that
# turns up) has to be pip-installed into the user's scripts/modules folder.
# That can fail for reasons entirely outside our control -- no network, a
# locked-down machine, a proxy, a read-only install -- so nothing here is
# allowed to raise.  Callers get None and decide how to cope.
def have_module(name):
    """Is the module importable right now, without importing it."""
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


_REQUIRE_TRIED = set()


def require(name, install=True, quiet=False):
    """Import `name`, offering to pip-install it once per session.

    Returns the module, or None if it is genuinely unavailable.  The install is
    attempted at most once per module per session: retrying a failed pip on
    every frame would hang the sim on a machine with no network.
    """
    import importlib
    try:
        return importlib.import_module(name)
    except Exception:
        pass

    if not install or name in _REQUIRE_TRIED:
        return None
    _REQUIRE_TRIED.add(name)

    if not quiet:
        print("MC: '%s' is not available, trying to install it..." % name)
    try:
        external_lib(module=name)
    except Exception as e:                       # pip can fail any number of ways
        print("MC: could not install '%s': %s" % (name, e))
        return None

    try:
        importlib.invalidate_caches()
        return importlib.import_module(name)
    except Exception as e:
        print("MC: '%s' still not importable after install: %s" % (name, e))
        return None


def install_package__(package_name):
    """
    Installs a Python package in Blender's Python environment.
    """
    python_exe = setup_pip_in_blender()
    
    print(f"Installing {package_name}...")
    subprocess.call([python_exe, "-m", "pip", "install", package_name])
    
    # Reload site to recognize newly installed packages
    site.addsitedir(site.getsitepackages()[0])

    
def get_installed_packages():
    """
    Returns a list of installed packages in Blender's Python environment.
    """
    python_exe = setup_pip_in_blender()
    result = subprocess.check_output([python_exe, "-m", "pip", "list"]).decode()
    return result


# Example usage:


if False:
    install_package("opencv-python")
    # Install a package
    install_package("matplotlib")
    install_package("scipy")
    print("\nInstalled packages:")
    print(get_installed_packages())

    install_package("open3d")
    install_package("networkx")
    install_package("sklearn")

    try:
        from scipy.spatial import KDTree
    except:
        install_package("scipy")
        from scipy.spatial import KDTree

def KDTree(*args, **kwargs):
    """scipy's KDTree, imported (and offered for install) on first use.

    The functions below that use trees are optional extras, so scipy is not a
    hard requirement: without it they raise here rather than the addon failing
    to import.  Replaced by the real class once it is available.
    """
    global KDTree
    need = ("scipy is needed for this "
            "(MC5 settings > Dependencies > Install scipy)")
    if require("scipy") is None:
        raise ImportError(need)
    try:
        from scipy.spatial import KDTree as _kd
    except Exception as e:
        raise ImportError("%s [%s]" % (need, e))
    KDTree = _kd
    return _kd(*args, **kwargs)


# END PIP STUFF ===================


# python
def redraw(sleep=0.0):
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)  # Update the viewport
    if sleep != 0.0:    
        time.sleep(sleep)  # Pause to see the changes


# python
def save_data(name='saved_data.py', var='some_variable', data={'key': [1,2,3]}, overwrite=True):
    """Saves a dictionary as a variable in a python file
    as a blender internal text file. Can later import
    module and call all data as global variables."""
    if name not in bpy.data.texts:
        bpy.data.texts.new(name)

    data_text = bpy.data.texts[name]

    m = json.dumps(data, sort_keys=True, indent=2)

    if overwrite:
        data_text.from_string(var + ' = ' + m)
        return

    # can also just add to the text
    data_text.cursor_set(-1, character=-1) # in case someone moves the cursor
    data_text.write(var + ' = ' + m)

    data_text.cursor_set(-1, character=-1) # add the new line or we can't read it as a python module
    data_text.write('\n')


# to save an external text in a blend file
def save_text_in_blend_file(path, file_name='my_text.py'):
    """Run this then save the blend file.
    file_name is the key blender uses to store the file:
    bpy.data.texts[file_name]"""
    t = bpy.data.texts.new(file_name)
    read = open(path).read()
    t.write(read)


# to import the text as a module from within the blend file
def get_internal_text_as_module(filename, key):
    """Load a module and return a dictionary from
    that module."""
    module = bpy.data.texts[filename].as_module()
    return module.points[key]


# python
def list_to_string(lst):
    return repr(lst)


def string_to_list(stringy_list):
    return literal_eval(stringy_list)


def code_line():
    """Prints the line of code where it's called"""
    frame = inspect.currentframe()
    calling_line = frame.f_back.f_lineno
    print("Called on line:", calling_line)


# python
def numpy_array_to_string(numpy_array):
    return ','.join(map(str, numpy_array.ravel()))


# python
def string_to_numpy_array(string, shape=None, dtype=np.float32):
    ar = np.fromstring(string, dtype=dtype, sep=',')
    if shape is None:
        shape = (ar.shape[0]//3, 3)
    ar.shape = shape
    return ar


# python
def fast_attribute(ob, obm=None):
    """Faster than list comprehension
    for pulling attributes"""
    if obm is None:
        obm = get_bmesh(ob)
    ed1 = np.array(obm.edges, dtype=object)
    attribute_iterator = (obj.index for obj in ed1)
    edi = np.fromiter(attribute_iterator, dtype=np.int32)
    return edi


# python
def flatten_list(li, depth=1):
    """Takes an Nd list and returns it flat"""
    for _ in range(depth):
        li = [i for j in li for i in j]
    return li


# python
def split_float(value):
    int_part = int(value)
    decimal = value - int_part
    return int_part, decimal


# python
def r_print(r, m=''):
    """Prints numpy arrays
    and lists rounded off."""
    print()
    print(m + " --------")
    if isinstance(r, list):
        for v in r:
            print(np.round(v, 3), m)
        return        
    print(np.round(r, 3), m)


# python
def py_from_object(ob, round=3):
    """Writes the verts and faces to this file
    when in blender"""
    np.set_printoptions(suppress=True)
    vc = len(ob.data.vertices)
    co = np.empty((vc, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())
    r = np.round(co, round)
    
    col = []
    for i in r:
        vco = []
        for c in i:
            vco.append(c)
        col.append(vco)
    
    f = [[v for v in f.vertices] for f in ob.data.polygons]
    return str(col), str(f)


# python
def py_from_object_example():    
    v, f = py_from_object(bpy.context.object)

    t = bpy.data.texts['py_from_object.py']
    t.cursor_set(line = 34)
    t.write('verts = ' + v)    
    t.cursor_set(line = 35)
    t.write('faces = ' + f)


# python
def read_python_script(name=None):
    """When this runs it makes a copy of this script
    and saves it to the blend file as a text
    Not a virus... well sort of like a virus"""

    p_ = pathlib.Path(inspect.getfile(inspect.currentframe()))
    py = p_.parts[-1]
    p = p_.parent.joinpath(py)
    try:
        o = open(p)
    except:
        p = p_.parent.joinpath(py) # linux or p1 (not sure why this is happening in p1)
        o = open(p)

    if name is None:
        name = 'new_' + py

    new = bpy.data.texts.new(name)

    r = o.read()
    new.write(r)


#=======================#
# WEB ------------------#
#=======================#
# web
def open_browser(link=None):
    import webbrowser

    if link == "paypal":
        # subscribe with paypal:
        link = "https://www.paypal.com/webapps/billing/plans/subscribe?plan_id=P-8V2845643T4460310MYYRCZA"
    if link == "gumroad":
        # subscribe with gumroad:
        link = "https://richcolburn.gumroad.com/l/dtnqq"
    if link == "patreon":
        # subscribe with patreon:
        link = "https://www.patreon.com/checkout/TheologicalDarkWeb?rid=23272750"
    if link == "donate":
        # paypal donate:
        link = "https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=4T4WNFQXGS99A"
    
    webbrowser.open(link)


### ==================== ###
#      VERTEX GROUPS       #
### ==================== ###
# vertex groups
def get_weights_fast_with_group_check(ob, vgroup_name, weights=None):
    """Fast extraction of vertex weights for a specific group"""
    vgroup_idx = ob.vertex_groups[vgroup_name].index
    
    if weights is None:    
        weights = np.zeros(len(ob.data.vertices), dtype=np.float32)
    
    for i, vert in enumerate(ob.data.vertices):
        for group in vert.groups:
            if group.group == vgroup_idx:
                weights[i] = group.weight
                break
    
    return weights


# vertex groups
def get_vertex_weights_fast_no_check(ob, vgroup_idx, weights=None):
    """Fastest - when ALL vertices are guaranteed to be in the group"""
    vgroup = ob.vertex_groups[vgroup_idx]
    
    for vert in ob.data.vertices:
        weights[vert.index] = vgroup.weight(vert.index)
    
    return weights


# vertex groups
def create_missing_groups_edit_mode(ob, bm, v_list, name, weight=1.0):
    """Only make group if it doesn't exist - Edit Mode version"""
    if name not in ob.vertex_groups:
        nvg = ob.vertex_groups.new(name=name)
        deform_layer = bm.verts.layers.deform.verify()
        
        for v_idx in v_list:
            bm.verts[v_idx][deform_layer][nvg.index] = weight
        
        return True
    return False


# vertex groups
def create_missing_groups(ob, v_list, name, weight=1.0):
    """Only make group if it doesn't exist"""
    if name not in ob.vertex_groups:
        nvg = ob.vertex_groups.new(name=name)
        nvg.add(v_list, weight, 'ADD')
        return True
    return False
    

# vertex groups
def assign_vert_group(ob, v_list, name, weight=1.0):
    """Does what you might imagine
    (makes me a sandwich)."""
    if name not in ob.vertex_groups:
        nvg = ob.vertex_groups.new(name=name)
    else:
        nvg = ob.vertex_groups[name]
    nvg.add(v_list, weight, 'ADD')


# vertex groups
def get_weight(ob, v, group):
    """Try to get single weight. If vert not in group
    retun 0.0"""
    vertex_group = ob.vertex_groups[group]
    try:    
        weight = vertex_group.weight(v)
    except:
        return 0.0
    return weight


# vertex groups
def assign_vertex_weights_edit(ob, weights, group_name):
    bm = bmesh.from_edit_mesh(ob.data)
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:    
        vg = ob.vertex_groups.new(name=group_name)
    deform_layer = bm.verts.layers.deform.verify()
    
    for i, vert in enumerate(bm.verts):
        if vg.index not in vert[deform_layer]:
            weights[i] = 0.0

        vert[deform_layer][vg.index] = weights[i]
    
    bmesh.update_edit_mesh(ob.data)


# vertex groups
def assign_vertex_weights(ob, weights, group_name):
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:    
        vg = ob.vertex_groups.new(name=group_name)
    for i, weight in enumerate(weights):
        vg.add([i], weight, 'REPLACE')


# vertex groups
def assign_vertex_weights_idx(ob, weights, group_name, idx):
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:    
        vg = ob.vertex_groups.new(name=group_name)
    for e, i, in enumerate(idx):
        if i in weights:    
            vg.add([e], weights[i], 'REPLACE')


# vertex groups
def assign_single(ob, group_name, vidx, weights):
    
    missing = []
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:    
        vg = ob.vertex_groups.new(name=group_name)
    for e, i, in enumerate(vidx):
        if e >= weights.shape[0]:
            missing += [e]
        else:
            vg.add([int(i)], weights[e], 'REPLACE')
    return missing    


# vertex groups
def write_vertex_group(ob, group_name, weights):
    """Set every vert's weight in one pass, creating the group if needed.
    `weights` is one value per vert, in vert order."""
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:
        vg = ob.vertex_groups.new(name=group_name)
    for i, w in enumerate(weights):
        vg.add([i], float(w), 'REPLACE')


# vertex groups
def assign_single_ref(ob, group_name, vidx, weights):
    if group_name in ob.vertex_groups:
        vg = ob.vertex_groups[group_name]
    else:    
        vg = ob.vertex_groups.new(name=group_name)
    for e, i, in enumerate(vidx):
        vg.add([e], weights[i], 'REPLACE')


# vertex groups
def get_weights(ob, verts, group):
    """Try to get list of weights. If vert not in group
    set that vert to 0.0"""
    vertex_group = ob.vertex_groups[group]
    weights = np.zeros(len(verts), dtype=np.float32)
    for e, v in enumerate(verts):
        try:    
            weights[e] = vertex_group.weight(v)
        except:
            weights[e] = 0.0
    return weights


# vertex groups
def get_vertex_weights(ob, group_name, default=1.0):
    """Get weights of the group. if it's not
    in the group set it to default"""
    if group_name not in ob.vertex_groups:
        ob.vertex_groups.new(name=group_name)
        v_list = np.arange(len(ob.data.vertices)).tolist()
        assign_vert_group(ob, v_list, group_name, weight=1.0)
        
    vertex_group = ob.vertex_groups[group_name]
    all_vertices = range(len(ob.data.vertices))
    weights = []

    for idx in all_vertices:
        try:
            weight = vertex_group.weight(idx)
            weights.append(weight)
        except RuntimeError:
            weights.append(default)

    vertex_weights = np.array(weights, dtype=np.float32)
    return vertex_weights


# vertex groups
def get_vertex_weights_overwrite(ob, group_name, default=1.0):
    """Get weights of the group. if it's not
    in the group set the weight of that
    vertex to default"""
    if group_name not in ob.vertex_groups:
        ob.vertex_groups.new(name=group_name)

    vertex_group = ob.vertex_groups[group_name]
    all_vertices = range(len(ob.data.vertices))
    weights = []

    for idx in all_vertices:
        try:
            weight = vertex_group.weight(idx)
            weights.append(weight)
        except RuntimeError:
            weights.append(default)

    vertex_weights = np.array(weights, dtype=np.float32)
    return vertex_weights


# vertex groups
def assign_default_weights(obj, weights, default_weight, group_name):
    """
    Assigns default weight to vertices not in the specified vertex group.
    Works in edit mode.
    
    Args:
        obj: Blender mesh object
        weights: numpy array of vertex weights (will be modified in-place)
        default_weight: weight value to assign to vertices not in group
        group_name: name of the vertex group to check
    """
    # Get or create the vertex group
    if group_name not in obj.vertex_groups:
        vg = obj.vertex_groups.new(name=group_name)
    else:
        vg = obj.vertex_groups[group_name]
    
    # Get the vertex group index
    vg_index = vg.index
    
    # Get bmesh from edit mode
    #bm = bmesh.from_edit_mesh(obj.data)
    bm = get_bmesh(obj)
    
    # Get the deform layer
    deform_layer = bm.verts.layers.deform.verify()
    
    # Iterate through all vertices
    for vert in bm.verts:
        vert_index = vert.index
        
        # Check if vertex is in the group
        is_in_group = vg_index in vert[deform_layer]
        
        # If not in group, assign default weight
        if not is_in_group:
            vert[deform_layer][vg_index] = default_weight
            try:    
                weights[vert_index] = default_weight
            except:
                print(vert_index, "was not in", weights)
            
            
    # Update the mesh
    set_bmesh(obj, bm)
    #bmesh.update_edit_mesh(obj.data)


# vertex groups
def vertex_groups_to_dict(ob, verts=None, offset=None):
    """Create a dictionary of groups for every vert
    or a list of specific verts if "vert" is
    an array or list of numbers.
    "offset" is for breaking verts out into a new object"""

    vgroup_names = {vgroup.index: vgroup.name for vgroup in ob.vertex_groups}

    if verts is None:
        vgroups = {v.index: [vgroup_names[g.group] for g in v.groups] for v in ob.data.vertices}
        return vgroups

    if offset is None:
        vgroups = {
            ob.data.vertices[v].index: [vgroup_names[g.group] for g in ob.data.vertices[v].groups] for v in verts
        }
        return vgroups

    vgroups = {
        ob.data.vertices[v].index - offset[v]: [vgroup_names[g.group] for g in ob.data.vertices[v].groups]
        for v in verts
    }
    return vgroups


#=======================#
# UV MAPS --------------#
#=======================#
def get_uv_index_from_3d(ob):
    """Creates a two dimensional list including where each vert
    occurs in the uv layers. Second dimension is N size so not numpy"""
    # figure out every index where the 3d verts occur in the uv maps
    obm = get_bmesh(ob)
    obm.verts.ensure_lookup_table()
    obm.faces.ensure_lookup_table()
    
    # currently works on a selected set of verts
    selected_verts = get_selected_verts(ob)
    sel_idx = np.arange(selected_verts.shape[0])[selected_verts]
    
    indexed_sum = []
    cum_sum = 0
    for i in obm.faces:
        indexed_sum.append(cum_sum)
        cum_sum += len(i.verts)
    
    v_sets = []
    for i in sel_idx:
        uv = []
        for f in obm.verts[i].link_faces:
            vidx = np.array([v.index for v in f.verts])
            idx = (np.arange(vidx.shape[0])[vidx == i])[0]
            uv.append(idx + indexed_sum[f.index])
        v_sets.append(uv)    
    # v_sets is now the uv index of each vert wherever it occurs in the uv map
    # v_sets[0] is vert zero and [uv[5], uv[390], uv[25]] or something like that
    # v_sets[0] looks like [5, 390, 16]
    return v_sets


#=======================#
# BMESH ----------------#
#=======================#

# bmesh
def get_ordered_loop(poly_line, edges=None, mask=None, obm=None):
    """Takes a bunch of verts and gives the order
    based on connected edges.
    Or, takes an edge array of vertex indices
    and gives the vertex order."""

    if edges is not None:
        e_idxer = np.arange(edges.shape[0])[mask]
        v = edges[mask][0][0]

        eid = e_idxer[np.any(v == edges[mask], axis=1)]
        mask[eid] = False
        le = edges[eid]
        if len(le) != 2:
            print("requires a continuous loop of edges")
            return
            
        ordered = [v]
        for i in range(len(poly_line.data.vertices)):
            #le = v.link_edges
            eid = np.any(v == edges, axis=1)
            mask[eid] = False
            le = edges[eid]
            #used += e_idxer[eid].tolist()
            if len(le) != 2:
                print("requires a continuous loop of edges")
                break

            ot1 = le[0][le[0] != v]
            ot2 = le[1][le[1] != v]
            v = ot1
            if ot1 in ordered[-2:]:    
                v = ot2
            if v == ordered[0]:
                break

            ordered += [v[0]]
            
        return ordered
        
    if obm is None:    
        obm = get_bmesh(poly_line, refresh=True)
    v = obm.edges[0].verts[0]
    le = v.link_edges

    if len(le) != 2:
        print("requires a continuous loop of edges")
        return
        
    ordered = [v.index]
    for i in range(len(poly_line.data.vertices)):
        le = v.link_edges

        if len(le) != 2:
            print("requires a continuous loop of edges")
            break

        ot1 = le[0].other_vert(v)
        ot2 = le[1].other_vert(v)
        v = ot1
        if ot1.index in ordered[-2:]:    
            v = ot2
        if v.index == ordered[0]:
            break

        ordered += [v.index]
    
    return ordered


# bmesh
def delete_faces(ob, obm=None, face_idx=[0], type=0):
    made_obm = False
    if obm is None:
        obm = get_bmesh(ob, refresh=True)
        made_obm = True
    faces = [obm.faces[i] for i in face_idx]
    context = ['FACES_ONLY', 'FACES'][type]
    bmesh.ops.delete(obm, geom=faces, context=context)
    if made_obm:
        set_bmesh(ob, obm)


# bmesh
def merge_verts(ob, margin=0.001, obm=None):
    """Uses the bmesh module to remove doubles."""
    if obm is None:
        obm = get_bmesh(ob, refresh=True)
    
    bmesh.ops.remove_doubles(obm, verts=obm.verts, dist=margin)
    obm.to_mesh(ob.data)

    ob.data.update()
    obm.clear()
    obm.free()


# bmesh
def face_from_verts(ob, vidx, obm=None):
        
    if obm is None:
        obm = get_bmesh(ob, refresh=True)
    
    verts = [obm.verts[i] for i in vidx]
    obm.faces.new(verts)

    if obm is None:
        set_bmesh(ob)


# bmesh
def check_faces(ob, vidx=None):
    if len(ob.data.polygons) == 0:
        obm = get_bmesh(ob, refresh=True)
        if vidx is None:    
            vidx = get_ordered_loop(ob, edges=None, obm=obm)
        face_from_verts(ob, vidx, obm=obm)
        set_bmesh(ob, obm)
        return True
    return False


# bmesh
def set_bmesh(ob, obm):
    if ob.data.is_editmode:
        bmesh.update_edit_mesh(ob.data)
    else:
        obm.to_mesh(ob.data)
        ob.data.update()


# bmesh
def unwrap_object(ob):
    
    mode = manage_modes()
    deselect_all()
    
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    manage_modes(mode)


# bmesh
def get_triobm(ob, respect_edit=False):
    """Takes an object, returns a triangle bmesh.
    Works in edit mode."""
    if respect_edit:
        if ob.data.is_editmode:
            obm = bmesh.from_edit_mesh(ob.data)
            bmesh.ops.triangulate(obm, faces=obm.faces)
            return obm

    obm = bmesh.new()
    obm.from_mesh(ob.data)
    bmesh.ops.triangulate(obm, faces=obm.faces)
    obm.edges.index_update()        # insurance: ops are not documented to
    obm.faces.index_update()        # keep element indices valid
    return obm


# bmesh
def _triangulated_bmesh(ob, use_prox=False):
    """A private, triangulated bmesh of `ob` with valid element indices.

    In edit mode get_bmesh() hands back the LIVE edit bmesh, and
    triangulating that triangulated the user's own mesh.  Work on a copy.

    The index_update() calls are insurance: bmesh ops are not documented to
    keep element indices valid.  (Blender 5.2's triangulate does keep them,
    measured.)
    """
    if use_prox:
        obm = get_bmesh_evaluated(ob)
    elif ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data).copy()
    else:
        obm = bmesh.new()
        obm.from_mesh(ob.data)
    bmesh.ops.triangulate(obm, faces=obm.faces[:])
    obm.verts.index_update()
    obm.edges.index_update()
    obm.faces.index_update()
    return obm


def get_tri_edges(ob, use_prox=False):
    triobm = _triangulated_bmesh(ob, use_prox)

    tridex = np.array([[t.verts[0].index, t.verts[1].index, t.verts[2].index] for t in triobm.faces], dtype=np.int32)
    tri_eidx = np.array([[e.verts[0].index, e.verts[1].index] for e in triobm.edges])
    tri_eidxer = np.array([e.index for e in triobm.edges], dtype=np.int32)
    tri_edge_idxer = np.array([[e.index for e in f.edges] for f in triobm.faces], dtype=np.int32)
    triobm.free()
    return tridex, tri_eidx, tri_eidxer, tri_edge_idxer


def get_edge_normal_data(ob, use_prox=False):
    triobm = _triangulated_bmesh(ob, use_prox)

    edge_to_normal_idxer = []
    edge_to_normal_counts = []
    edge_to_normal_add_idxer = []
    
    for i, ed in enumerate(triobm.edges):
        lf = [f.index for f in ed.link_faces]
        edge_to_normal_idxer += lf
        edge_to_normal_counts += [len(lf)]    
        edge_to_normal_add_idxer += [i] * len(lf)
    
    e1 = np.array(edge_to_normal_idxer, dtype=np.int32)
    e2 = np.array(edge_to_normal_counts, dtype=np.float32)
    e3 = np.array(edge_to_normal_add_idxer, dtype=np.int32)
    triobm.free()
    return e1, e2, e3

def get_edge_normals(normals, edge_normals, data):
    idxer = data["edge_to_normal_idxer"]
    adder = data["edge_to_normal_add_idxer"]
    
    edge_normals[:] = 0.0
    
    np.add.at(edge_normals, adder, normals[idxer])
    edge_normals /= np.linalg.norm(edge_normals, axis=1)[:, None]


# bmesh
def get_mesh_bmesh(mesh):
    obm = bmesh.new()
    obm.from_mesh(mesh)
    return obm


# bmesh
def get_bmesh_evaluated(ob):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obm = bmesh.new()
    obm.from_object(ob, depsgraph)    
    return obm

    
# bmesh
def get_bmesh(ob=None, refresh=False, mesh=None, copy=False):
    """gets bmesh in editmode or object mode
    by checking the mode"""
    if ob.data.is_editmode:
        obm = bmesh.from_edit_mesh(ob.data)
        return obm
    obm = bmesh.new()
    m = ob.data
    if mesh is not None:
        m = mesh

    obm.from_mesh(m)
    if refresh:
        obm.verts.ensure_lookup_table()
        obm.edges.ensure_lookup_table()
        obm.faces.ensure_lookup_table()

    if copy:
        return obm.copy()

    return obm


# bmesh
def delete_verts_by_bool(ob, vboo):
    """Deletes verts by bool, imagine that."""
    obm = get_bmesh(ob)
    verts_to_remove = [v for i, v in enumerate(obm.verts) if vboo[i]]
    bmesh.ops.delete(obm, geom=verts_to_remove, context='VERTS')
    obm.to_mesh(ob.data)
    ob.data.update()
    obm.free()
    return ob


# bmesh
def delete_in_box(box, obs, delete_inside=True):
    """Delets points inside a box that can
    be rotated in object mode."""
#    M = np.array(((1.0, 0.0, 0.0, 0.0),
#         (0.0, 1.0, 0.0, 0.0),
#         (0.0, 0.0, 1.0, 0.0),
#         (0.0, 0.0, 0.0, 1.0)))
    
    bco = get_co(box)
    bmin, bmax = get_bounds(bco)                

    for ob in obs:
        M = np.array(ob.matrix_world)
        co = get_co(ob)
        local_matrix = np.linalg.inv(box.matrix_world) @ M
        local_co = co @ local_matrix[:3, :3].T
        local_co += local_matrix[:3, 3]    
        in_box = co_in_bounds(local_co, bmin, bmax)
        if delete_inside:    
            delete_verts_by_bool(ob, in_box)
        else:
            delete_verts_by_bool(ob, ~in_box)


#bmesh
def bmesh_recalculate_normals(ob, prox=False, write=False):
    """Uses bmesh recalc normals.
    Returns the bmesh and a bool for
    normals that were flipped."""
    if prox:
        ob = prox_object(ob)
    obm = get_bmesh(ob)

    norms = np.array([f.normal for f in obm.faces])
    bmesh.ops.recalc_face_normals(obm, faces=obm.faces)
    norms2 = np.array([f.normal for f in obm.faces])
    compare = compare_vecs(norms, norms2)
    flipped = compare < 0.0
        
    if write:
        obm.to_mesh(ob.data)
        ob.data.update()
    
    return flipped, obm


# bmesh
def get_tridex(ob, teidx=False, tmesh=False, tobm=None, refresh=False, cull_verts=None, free=False):
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get.
    Return tridex and triobm"""

    if tobm is None:
        tobm = get_bmesh(ob)
        if cull_verts is not None:
            tobm.verts.ensure_lookup_table()
            cv = [tobm.verts[v] for v in cull_verts]
            bmesh.ops.delete(tobm, geom=cv, context='VERTS')

    bmesh.ops.triangulate(tobm, faces=tobm.faces)
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=D_TYPE_I)
    me.polygons.foreach_get('vertices', tridex.ravel())
    three_edges = np.empty((len(tobm.faces), 3), dtype=D_TYPE_I)
    for e, f in enumerate(tobm.faces):
        eid = [ed.index for ed in f.edges]
        three_edges[e] = eid

    if free:
        edge_normal_keys = []
        for e in tobm.edges:
            if len(e.link_faces) == 2:
                e_key = [e.link_faces[0].index, e.link_faces[1].index]
            elif len(e.link_faces) == 1:
                e_key = [e.link_faces[0].index, e.link_faces[0].index]
            else:
                e_key = [-1, -1]    
            edge_normal_keys += [e_key]
                
        eidx = np.array(me.edge_keys, dtype=D_TYPE_I)
        bpy.data.meshes.remove(me)
        tobm.free()
        return tridex, eidx, three_edges, np.array(edge_normal_keys, dtype=D_TYPE_I)
    
    if tmesh:
        return tridex, tobm, me
    if teidx:
        eidx = np.array(me.edge_keys, dtype=D_TYPE_I)
        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        return tridex, eidx, tobm

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    if refresh:
        tobm.verts.ensure_lookup_table()
        tobm.edges.ensure_lookup_table()
        tobm.faces.ensure_lookup_table()

    return tridex, tobm, None


# bmesh
def get_tridex_2(ob, mesh=None): # faster than get_tridex()
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get. Faster than get_tridex()"""

    if mesh is not None:
        tobm = bmesh.new()
        tobm.from_mesh(mesh)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
        me = bpy.data.meshes.new('tris')
        tobm.to_mesh(me)
        p_count = len(me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        me.polygons.foreach_get('vertices', tridex.ravel())

        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        if ob == 'p':
            return tridex, tobm

        tobm.free()
        return tridex

    if ob.data.is_editmode:
        ob.update_from_editmode()

    tobm = bmesh.new()
    tobm.from_mesh(ob.data)
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=np.int32)
    me.polygons.foreach_get('vertices', tridex.ravel())

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    return tridex, tobm


#=======================#
# SELECT ---------------#
#=======================#

# select
def get_selected_verts(ob):
    vidx = np.arange(len(ob.data.vertices))
    sel = np.zeros(vidx.shape[0], dtype=bool)
    ob.data.vertices.foreach_get("select", sel)
    return vidx[sel]
    

# select
def select_verts(ob, vert_bool=None):
    ed = np.zeros(len(ob.data.edges), dtype=bool)
    ob.data.edges.foreach_set("select", ed)
    pg = np.zeros(len(ob.data.polygons), dtype=bool)
    ob.data.polygons.foreach_set("select", pg)
    if vert_bool is not None:
        ob.data.vertices.foreach_set("select", vert_bool)
    ob.data.update()


# select
def select_edges(ob, edge_bool=None):
    ve = np.zeros(len(ob.data.vertices), dtype=bool)
    ob.data.vertices.foreach_set("select", ve)
    pg = np.zeros(len(ob.data.polygons), dtype=bool)
    ob.data.polygons.foreach_set("select", pg)
    if edge_bool is not None:
        ob.data.edges.foreach_set("select", edge_bool)
    ob.data.update()
    

# select    
def select_faces(ob, face_bool=None):
    ve = np.zeros(len(ob.data.vertices), dtype=bool)
    ob.data.vertices.foreach_set("select", ve)
    pg = np.zeros(len(ob.data.polygons), dtype=bool)
    ob.data.polygons.foreach_set("select", pg)
    if face_bool is not None:
        ob.data.polygons.foreach_set("select", face_bool)
    ob.data.update()


# select
def deselect(ob, sel=None, type='vert'):
    """Deselect all then select something"""
    x = np.zeros(len(ob.data.vertices), dtype=bool)
    y = np.zeros(len(ob.data.edges), dtype=bool)
    z = np.zeros(len(ob.data.polygons), dtype=bool)

    ob.data.vertices.foreach_set('select', x)
    ob.data.edges.foreach_set('select', y)
    ob.data.polygons.foreach_set('select', z)
    
    if sel is not None:    
        if type == 'vert':    
            x[sel] = True
            ob.data.vertices.foreach_set('select', x)
        if type == 'edge':
            y[sel] = True
            ob.data.edges.foreach_set('select', y)
        if type == 'face':
            z[sel] = True
            ob.data.polygons.foreach_set('select', z)
    ob.data.update()


# select
def set_vert_select_edit_mode(ob, obm=None, verts=None):
    """Updates selection in edit mode using
    bmesh. "verts" is a bool array matching 
    the number of verts in the bmesh."""    
    if obm is None:
        obm = get_bmesh(ob, refresh=True)
    for i, vert in enumerate(obm.verts):
        print(verts[i], "verts i here!!!")
        vert.select = bool(verts[i].item())

    obm.select_flush_mode()
    bmesh.update_edit_mesh(ob.data)


# select
def shrink_isolated(obm, sel):
    """Optimized shrink selection using set comprehension"""
    for i in range(len(obm.verts)):
        if sel[i]:    
            v = obm.verts[i]
            otv = [e.other_vert(v).index for e in v.link_edges]
            count = np.count_nonzero(sel[otv])
            if count < 3:
                sel[i] = False


# select
def grow(obm, iters=4, sel=None, compute_steps=False):
    """Uses bmesh to grow selection.
    Has the option to return an array of numbers
    for the steps it took to get to each vert.
    Verts already selected will have a value of
    zero in steps: 0. Verts completely ignored
    will have a value of negative one: -1 """
    vc = len(obm.verts)
    if sel is None:
        sel = np.array([v.select for v in obm.verts])
    otvs = set()
    
    if compute_steps:
        steps = np.zeros(vc, dtype=np.int32)
        steps[:] = -1
        steps[sel] = 0
    step_count = 1
    for it in range(iters):
        
        if np.all(sel):
            if compute_steps:
                print("ducked out early because there were no more sandwiches")
                return sel, steps
            return sel
        
        new = np.zeros(vc, dtype=bool)
        for v in obm.verts:
            if sel[v.index]:    
                lv = [le.other_vert(v).index for le in v.link_edges]    
                otvs.update(lv)
    
        npotv = np.fromiter(otvs, dtype=int)
        if compute_steps:    
            new = npotv[~sel[npotv]]
            steps[new] = step_count
            step_count += 1
                
        sel[npotv] = True
    
    if compute_steps:
        return sel, steps    
        
    return sel


#=======================#
# BLENDER OBJECTS ------#
#=======================#

# blender objects
def clean_meshes():
    for m in bpy.data.meshes:
        if m.users == 0:
            bpy.data.meshes.remove(m)


# blender objects
def attribute_object(ob, use_prox=True):
    """Using this to create a reference object for flipped
    faces in the geometry nodes."""
    if use_prox:    
        ob = prox_object(ob)
    flipped, obm = bmesh_recalculate_normals(ob)
    
    hidden_mesh = bpy.data.meshes.new(ob.name + "_flip_mesh")
    obm.to_mesh(hidden_mesh)
    hidden_object = bpy.data.objects.new(ob.name + "_flip_object", hidden_mesh)
    
    hidden_object.data.update()
    hidden_object.data.attributes.new('flip_faces', "BOOLEAN", "FACE")
    hidden_object.data.attributes['flip_faces'].data.foreach_set('value', flipped)
    return hidden_object


# blender objects
def create_empty(location, name="e", empty_type='SPHERE'):
    """Creates an empty and links it to the scene."""
    bpy.ops.object.empty_add(type=empty_type, location=location)
    empty = bpy.context.object
    empty.name = name
    empty.show_axis = True
    if empty.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(empty)
    return empty


# blender objects
def create_camera(location=None, rotation=None, scale=None, name="cam_ur_uh"):
    """Create a camera object then throw it across the
    room hoping it doesn't hit anyone in the mouth.'"""
    if name in bpy.data.objects:
        print(f"Camera {name} was already present")
        return bpy.data.objects[name]
    
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    if location is not None:    
        cam_obj.location = location
    if rotation is not None:    
        cam_obj.rotation_euler = rotation
    if scale is not None:
        cam_obj.scale = np.array([scale, scale, scale])
    bpy.context.collection.objects.link(cam_obj)
    
    return cam_obj    
    

# geometry
def co_in_bounds(co, bmin, bmax):
    """Returns a bool array for verts
    in the box"""
    booler = np.ones(co.shape, dtype=bool)
    booler = (co < bmin) | (co > bmax)
    out = np.any(booler, axis=1)
    return ~out


# geometry
def sub_points_import(box, co):#, colors=None, rows=None, cols=None, intensity=None):
    
    M = np.array(((1.0, 0.0, 0.0, 0.0),
         (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0),
         (0.0, 0.0, 0.0, 1.0)))
    
    local_matrix = np.linalg.inv(box.matrix_world) @ M
    local_co = co @ local_matrix[:3, :3].T
    local_co += local_matrix[:3, 3]    
    
    bco = get_co(box)
    bmin, bmax = get_bounds(bco)                
    in_box = co_in_bounds(local_co, bmin, bmax)
    return in_box
    

def sub_mesh(ob=None, bounds_ob=None, inside=False, points=False):
    """Unless specified, uses the active mesh
    and any selected boxes. Expects roatations
    for the boxes only in object mode, not
    the bounds objects' elements. So don't
    rotate verts in edit mode.,
    """
    
    if ob is None:
        ob = bpy.context.object
        
    if bounds_ob is None:
        sel_obs = [sob for sob in bpy.data.objects if sob.select_get()]
        bounds_obs = [bob for bob in sel_obs if bob != ob]
    
    if ob is None:
        popup_error("No active object... your mom's an active object.")
        return
        
    if len(bounds_obs) == 0:
        popup_error("Selected objects empty. So's your face.")
        return
    
    co = get_co(ob)
    col = get_color_attributes(ob, color=0)
    M = ob.matrix_world
    
    booler = np.zeros(co.shape[0], dtype=bool)
    
    for bob in bounds_obs:
        lco = co_to_matrix_space(co, ob, bob)
        bco = get_co(bob)
        bmin, bmax = get_bounds(bco)                
        in_box = co_in_bounds(lco, bmin, bmax)
        booler[in_box] = True
        
    if inside:
        booler = ~booler
    
    if points:
        co = co[booler]
        new_ob = ob_from_py_data(co, faces=[], edges=[], name=f"{ob.name}_sub_mesh")
        set_color_attributes(new_ob, col[booler], color=0)
        new_ob.data.update()
        new_ob.matrix_world = M    
        return new_ob
    
    if not points:
        f_type = len(ob.data.polygons[0].vertices) # triangles or quads
        f_verts = np.zeros((len(ob.data.polygons), f_type), dtype=np.int32)
        ob.data.polygons.foreach_get("vertices", f_verts.ravel())
    
        face_bool = booler[f_verts]
        faces_in = np.all(face_bool, axis=1) # or use all if we dont want faces party in
        f_verts_in = f_verts[faces_in]
        cboo = np.zeros(co.shape[0], dtype=bool)
        cboo[f_verts_in] = True
        new_idx = np.cumsum(cboo) - 1
        new_co = co[cboo]
    
    new_faces = new_idx[f_verts_in]
    new_ob = ob_from_py_data(new_co, new_faces, edges=[], name=f"{ob.name}_sub_mesh")
    set_color_attributes(new_ob, col[cboo], color=0)
    new_ob.data.update()
    new_ob.matrix_world = M        


# blender objects
def ob_from_py_data(co, faces, edges=[], name="new mesh"):
    """Simple object from numpy arrays or lists."""
    mesh = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    mesh.from_pydata(co, edges, faces)
    return ob


# ====== C++ ====== #
import ctypes
def build_face_vert_csr(obm):
    """Build the flat CSR layout from a bmesh's faces."""
    face_verts_list = []
    face_starts = [0]
 
    for face in obm.faces:
        vidx = [v.index for v in face.verts]
        face_verts_list.extend(vidx)
        face_starts.append(face_starts[-1] + len(vidx))
 
    face_verts = np.array(face_verts_list, dtype=np.int32)
    face_verts = np.ascontiguousarray(face_verts, dtype=np.int32)
    
    face_starts = np.array(face_starts, dtype=np.int32)
    face_starts = np.ascontiguousarray(face_starts, dtype=np.int32)

    return face_verts, face_starts


def load_compute_face_centers(dll_path):
    lib = ctypes.CDLL(dll_path)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
 
    lib.compute_face_centers.argtypes = [
        c_float_p,  # co
        c_int_p,    # face_verts
        c_int_p,    # face_starts
        ctypes.c_int,  # num_faces
        c_float_p,  # out_centers
    ]
    lib.compute_face_centers.restype = None
    return lib


# ====== C++ ====== #
def compute_face_normals_cpp(lib, co, face_verts, face_starts, num_faces, out_normals):

    #co = np.ascontiguousarray(co.reshape(-1), dtype=np.float32)
 
    lib.compute_face_normals(
        co.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        face_verts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        face_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(num_faces),
        out_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
 
    return out_normals.reshape(num_faces, 3)


def compute_face_centers_cpp(lib, co, face_verts, face_starts, num_faces, out_centers):

    #co = np.ascontiguousarray(co.reshape(-1), dtype=np.float32)
 
    lib.compute_face_centers(
        co.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        face_verts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        face_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(num_faces),
        out_centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
 
    return out_centers.reshape(num_faces, 3)
# ====== C++ ====== #


# geometry
def get_face_means(fco, add_idxer, accumulator, sum_multiplier):
    np.add.at(accumulator, add_idxer, fco)
    

# geometry    
def get_face_centers(ob, means=None):
    # to test the speed we will need a proxy for the shape key if using this with cloth stuff
    if means is None:    
        fc = len(ob.data.polygons)
        means = np.empty((fc, 3), dtype=np.float32)
    ob.data.polygons.foreach_get('center', means.ravel())
    return means


# geometry
def get_face_centers_mesh(mesh, means=None):
    # to test the speed we will need a proxy for the shape key if using this with cloth stuff
    if means is None:    
        fc = len(mesh.polygons)
        means = np.empty((fc, 3), dtype=np.float32)
    mesh.polygons.foreach_get('center', means.ravel())
    return means
    

# geometry
def get_vertex_normals_np(ob, co, tridex=None):
    if tridex is None:    
        tridex = get_tridex_3(ob)
    norms = np.zeros_like(co)
    
    v1 = co[tridex[:, 1]] - co[tridex[:, 0]]
    v2 = co[tridex[:, 2]] - co[tridex[:, 0]]
    cross = fastest_cross_product(v1, v2)
    tri_normals = u_vecs(cross)
        
    #for i in range(3):
        #np.add.at(norms, tridex[:, i], vertex_normals)

    # do this instead?
    np.add.at(norms, tridex.ravel(), np.repeat(tri_normals, 3, axis=0))
    norms = u_vecs(norms)

    return norms
    

# get set
def get_tridex_3(ob):
    """100x faster than get_tridex_2 or 3"""
    #ob.data.calc_loop_triangles()  # Ensure the mesh is triangulated
    triangles = np.empty((len(ob.data.loop_triangles), 3), dtype=np.int32)
    ob.data.loop_triangles.foreach_get('vertices', triangles.ravel())
    return triangles


def get_extended_tridex(ob, steps=1):
    """For every vertex, find traingles and verts in triangles
    using vertex and verts in triangles N steps away"""
    tridex = get_tridex_3(ob)
    tridexer = np.arange(tridex.shape[0])
    vc = len(ob.data.vertices)        
    lookup = {}
    for i in range(vc):
        tris_in = np.any(i == tridex, axis=1)
        lookup[i] = tridexer[tris_in]
    
    return lookup
    
    
    """so... I could cull points based on radius
    using cpoe and finding edges where cpoe
    is shorter than radius.
    
    could start simple by auto-setting radius
    using cpoe. Needs to be the cpoe of edges
    in tridex.
    
    Bad topology will create problems but this
    could also be used to select bad topology.
    Or... we could identify the average min distance
    and put points that are closer in the lookup.
    
    Need to test if there is a way to use the lookup
    that is reasonably fast."""
    
    

# debug
def testing_get_tridex():
    T = time.time()
    tris = get_tridex_3(bpy.context.object)
    print(time.time() - T)
    T = time.time()
    get_tridex_2(bpy.context.object)
    print(time.time() - T)

    ob = bpy.context.object
    vc = len(ob.data.vertices)
    co = np.empty((vc, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())

    ob_from_py_data(co, tris, name="new mesh")


# get set
def edges_from_tridex_set(tridex):
    """Tested slower than edges from tridex"""
    edges = set()
    for v in tridex:
        # Get the vertex indices of the triangle
        # Add the 3 edges of the triangle (ensure order is consistent to avoid duplicates)
        edges.add(tuple(sorted((v[0], v[1]))))
        edges.add(tuple(sorted((v[1], v[2]))))
        edges.add(tuple(sorted((v[2], v[0]))))

    # Convert the set of edges to a NumPy array
    edges = np.array(list(edges), dtype=np.int32)
    return edges


# get set
def edges_from_tridex(tridex):
    """Tested faster than edges from tridex set"""
    edges = np.vstack([
        tridex[:, [0, 1]],  # Edge (v0, v1)
        tridex[:, [1, 2]],  # Edge (v1, v2)
        tridex[:, [2, 0]],  # Edge (v2, v0)
    ])

    # Sort the edges to ensure (v0, v1) and (v1, v0) are treated as the same
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges using np.unique
    edges = np.unique(edges, axis=0)
    return edges


# blender objects
def generate_edge_mesh(ob=None, triangulate=False):
    """Create a new mesh using only the edges.
    Optionally triangulate the mesh."""
    if ob is None:    
        ob = bpy.context.object
    vc = len(ob.data.vertices)
    co = np.empty((vc, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', co.ravel())

    if triangulate:
        tris = get_tridex_3(ob)
        edges = edges_from_tridex(tris)
    else:
        edges = ob.data.edge_keys
    
    ob_from_py_data(co, faces=[], edges=edges, name="new mesh")


# bmesh
def nearest_points_on_bmesh(coords, obm, return_all=False):
    """Uses blender closest_point... with
    a bmesh."""
    
    bvh = mu.bvhtree.BVHTree.FromBMesh(obm)
            
    faces = []
    locs = []
    norms = []
    dists = []
    
    for co in coords:
        loc, norm, face_index, dist = bvh.find_nearest(co)
        faces += [face_index]
        locs += [loc]
        if return_all:    
            norms += [norm]
            dists += [dist]
    if return_all:
        return faces, locs, norms, dists

    return faces, locs


#=======================#
# SHAPE KEYS -----------#
#=======================#
# shape keys
def manage_shapes(ob, shapes=None, values=None):
    if shapes is None:
        shapes = ["Basis", "Current"]
    
    # In case shapes is a single string
    if isinstance(shapes, str):
        shapes = [shapes]
    
    if ob.data.shape_keys is None:
        for sh in shapes:
            ob.shape_key_add(name=sh)
    
    for sh in shapes:
        if sh not in ob.data.shape_keys.key_blocks:
            ob.shape_key_add(name=sh)
            
    if values:
        keys = [ob.data.shape_keys.key_blocks[shape] for shape in shapes] 
        for e, v in enumerate(values):
            keys[e].value = v
        
        
#=======================#
# HANDLERS -------------#
#=======================#
def install_timer(function, clear=False):
    if bpy.app.timers.is_registered(function):
        bpy.app.timers.unregister(function)

    if clear:
        print('cleared timer:', function.__name__)
        return
        
    bpy.app.timers.register(function, persistent=True)
    print("installed timer:", function.__name__)


#=======================#
# DEBUG ----------------#
#=======================#
# debug
def add_debug_shape(ob, co, name="debug"):
    ob.shape_key_add(name="Basis")
    ob.shape_key_add(name=name)
    ob.data.shape_keys.key_blocks[name].data.foreach_set('co', co.ravel())
    ob.data.update()


# debug
def oops(self, context):
    """Blender convention for popup error messages."""
    print()
    return


# debug
def popup_error(msg, icon='ERROR'):
    """Simple way to make an error popup. Just put in
    a message."""
    def oops(self, context):
        return
    bpy.context.window_manager.popup_menu(oops, title=msg, icon=icon)


# debug
CT = None
def timer():
    return
    
    global CT
    if CT is None:
        CT = time.time()
        return
    frame = inspect.currentframe()
    calling_line = frame.f_back.f_lineno
    print()
    print("line:", calling_line)
    print(time.time() - CT)
    CT = time.time()
    print()


#=======================#
# BLENDER OBJECTS ------#
#=======================#
def deselect_all():
    """Set the select state of all blender
    objects to False."""
    for ob in bpy.data.objects:
        ob.select_set(False)


# Character Physics
def validate_references(ph):
    try:            
        ph.physics_rig.name
        ph.pose_target.name
        ph.mix_mesh.name
    except ReferenceError:
        print("atempting to retrieve id")
        print(ph.physics_rig_id, "ph.physics_rig_id")
        ph.physics_rig = retrieve_ob(ph.physics_rig_id)
        ph.pose_target = retrieve_ob(ph.pose_target_id)
        ph.mix_mesh = retrieve_ob(ph.mix_mesh_id)            
    except AttributeError:
        print("attempting in attribute error")
        ph.physics_rig = retrieve_ob(ph.physics_rig_id)
        ph.pose_target = retrieve_ob(ph.pose_target_id)
        ph.mix_mesh = retrieve_ob(ph.mix_mesh_id)


# blender objects
def set_id_key(ob):
    key = np.max([ob.CP_props.id_key for ob in bpy.data.objects]) + 1
    ob.CP_props.id_key = key
    return key
    
    
# blender objects
def retrieve_ob(key):
    obs = [ob for ob in bpy.data.objects if ob.CP_props.id_key == key]
    if len(obs) == 1:
        return obs[0]
    

# blender objects
def check_object(ob):
    return ob.name in bpy.context.scene.objects


# blender objects
def make_parent(ob1, ob2):
    """Make ob1 the child of ob2"""
    bpy.context.view_layer.update()
    ob1.parent = ob2
    ob1.matrix_parent_inverse = ob2.matrix_world.inverted()


#### armature ####
def get_bone_head(ar, head=None):
    """Return Nx3 coordinates for bone head"""
    if head is None:
        head = np.empty((len(ar.pose.bones), 3), dtype=np.float32)
    ar.pose.bones.foreach_get('head', head.ravel())
    return head


#### armature ####
def get_bone_tail(ar, tail=None):
    if tail is None:
        tail = np.empty((len(ar.pose.bones), 3), dtype=np.float32)
    ar.pose.bones.foreach_get('tail', tail.ravel())
    return tail


# blender objects
def link_armature(rig=None):
    # Create a new armature object
    scale = 1
    if rig:
        M = rig.matrix_world.copy()
        head = get_bone_head(rig)
        tail = get_bone_tail(rig)
        dif = tail - head
        dist = measure_vecs(dif)
        scale = np.mean(dist)        
        
    mode = manage_modes()
    armature = bpy.data.armatures.new("Armature")
    armature_obj = bpy.data.objects.new("Armature", armature)
    
    # Link armature object to the current scene
    bpy.context.collection.objects.link(armature_obj)
    
    # Add a single bone to the armature
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.bone_primitive_add(name="Bone")

    edit_bone = armature_obj.data.edit_bones[0]
    edit_bone.head = (0, 0, 0)
    edit_bone.tail = (0, 0, scale)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    manage_modes(mode)
        
    if rig:
        armature_obj.matrix_world = M
    
    return armature_obj


def remove_object(ob):
    mesh = ob.data
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(mesh)


def link_mesh_simple(verts, edges, faces, name='name', ob=None):
    """Generate and link a new object from pydata.
    If object already exists replace its data
    with a new mesh and delete the old mesh."""
    if ob is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, edges, faces)
        mesh.update()
        mesh_ob = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(mesh_ob)
        return mesh_ob
    
    mesh_ob = ob
    old = ob.data
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    mesh_ob.data = mesh
    bpy.data.meshes.remove(old)
    return mesh_ob


# blender objects
def link_mesh(verts, edges, faces, name='name', ob=None, preserve_keys=True, check_ob=True):
    """Generate and link a new object from pydata.
    If object already exists replace its data
    with a new mesh and delete the old mesh."""

    # check if object exists
    if check_ob:
        if name in bpy.data.objects:
            ob = bpy.data.objects[name]
            try:    
                bpy.context.collection.objects.link(ob)
            except:
                print("mesh already linked to scene")
            
            if preserve_keys:
                keys = {}
                active_idx = ob.active_shape_key_index
                if ob.data.shape_keys:
                    for k in ob.data.shape_keys.key_blocks:
                        keys[k.name] = get_shape_co_mode(ob, co=None, key=k.name)
        
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()

    if ob is None:
        ob = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(ob)
        return ob

    old = ob.data
    ob.data = mesh

    if preserve_keys:
        v_count = len(mesh.vertices)
        for k, v in keys.items():
            if v.shape[0] == v_count:
                ob.shape_key_add(name=k)
                ob.data.shape_keys.key_blocks[k].data.foreach_set("co", v.ravel())
        ob.data.update()
        ob.active_shape_key_index = active_idx
        
    bpy.data.meshes.remove(old)
    return ob


# blender objects
def new_empty(display_type, name, world_matrix=None, scene=None):
    if display_type not in {'PLAIN_AXES', 'SINGLE_ARROW', 'CIRCLE', 'CUBE', 'SPHERE', 'CONE', 'ARROWS'}:
        print(f"Invalid display type: {display_type}")
        return

    empty_object = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty_object)

    if scene is None:
        scene = bpy.context.scene
    #scene.collection.objects.link(empty_object)
    empty_object.empty_display_type = display_type
    if world_matrix:    
        empty_object.matrix_world = world_matrix

    return empty_object


def manage_modes(modes=None):
    """Check the modes and select state and stuff then, save that to dict.
    If dict is provided, set based on stored data. Also ensures the safetey
    of small reptilian squirrel hybrids if they are within 200 feet."""
    if modes:
            
        if modes['active']:    
            bpy.context.view_layer.objects.active = bpy.data.objects[modes['active']]
        else:
            bpy.context.view_layer.objects.active = None
        
        for k, v in modes['selected'].items():
            bpy.data.objects[k].select_set(v)
        
        ob = bpy.context.object
        if ob:
            print(modes['mode'], "What modes???")
            bpy.ops.object.mode_set(mode=modes['mode'])
            if ob.type == 'ARMATURE':
                for i in range(len(ob.pose.bones)):
                    ob.pose.bones[i].bone.select = modes['bone_select'][i]    
        return
    
    ob = bpy.context.object
    modes = {}
    if ob:    
        modes['active'] = ob.name
    else:
        modes['active'] = None
    
    modes['selected'] = {}
    for oby in bpy.context.view_layer.objects:
        modes['selected'][oby.name] = oby.select_get()
    modes['mode'] = bpy.context.mode

    if bpy.data.objects[modes['active']]:
        if ob.type == 'ARMATURE':
            modes["bone_select"] = [b.bone.select for b in ob.pose.bones]
    return modes


#=======================#
# BLENDER UI -----------#
#=======================#
# Enum:
def enum_get_v_groups(self, context):
    """ Enum Prop Function """
    ob = bpy.context.scene.LG_props.grid_object
    if ob:
        return [(group.name, group.name, "") for group in ob.vertex_groups]
    else:
        return []


#=======================#
# GEOMETRY -------------#
#=======================#
def detect_changes(ob, vc, ec, fc):
    
    if vc != len(ob.data.vertices):
        return True
    if ec != len(ob.data.edges):
        return True
    if fc != len(ob.data.polygons):
        return True
    
    return False

    
def detect_changes_mesh(mesh, vc, ec, fc):
    
    nvc = len(mesh.vertices)
    nec = len(mesh.edges)
    nfc = len(mesh.polygons)
    
    return np.any([nvc != vc, nec != ec, nfc != fc])
#    change = False
#    
#    if nvc != vc:
#        change = True
#    if nec != ec:
#        change = True
#    if nfc != fc:
#        change = True
#        
#    return change    
    

# ===== NUMPY ===== #
def get_common_pair_mask(current_pairs, previous_pairs):
    """Takes two Nx2 numpy arrays with similar or different N-counts.
    Returns a mask matching the shape of the first array for pairs
    that occur in the second array."""
    dtype = np.dtype([('vertex', current_pairs.dtype), ('triangle', current_pairs.dtype)])
    
    # View as 1D structured arrays
    current_view = current_pairs.view(dtype).ravel()
    previous_view = previous_pairs.view(dtype).ravel()
    
    # Compute mask: True if current pair exists in previous
    mask = np.isin(current_view, previous_view)
    
    return mask


# ===== NUMPY ===== #
def get_common_pair_mask_2(current_vertices, current_triangles, previous_vertices, previous_triangles):
    """Takes four Nx1 numpy arrays. The first two and last two need to match
    N-counts but the n-count of the first two doesn't have to match the n-count
    of the second two.
    Returns a mask matching the shape of the first array for pairs
    that occur in the second array."""    
    
    current_pairs = np.column_stack((current_vertices, current_triangles))
    previous_pairs = np.column_stack((previous_vertices, previous_triangles))
    
    # Define structured dtype based on the array dtype (assumes both columns same dtype)
    dtype = np.dtype([('vertex', current_pairs.dtype), ('triangle', current_pairs.dtype)])
    
    # View as 1D structured arrays
    current_view = current_pairs.view(dtype).ravel()
    previous_view = previous_pairs.view(dtype).ravel()
    
    # Compute mask: True if current pair exists in previous
    mask = np.isin(current_view, previous_view)
    
    return mask


# ===== GEOMETRY ===== #
def closest_point_on_mesh(ob, location):
    """Blender internal CPOM"""
    location = ob.matrix_world.inverted() @ location
    hit, hit_location, hit_normal, _ = ob.closest_point_on_mesh(location)
    w_hit = ob.matrix_world @ hit_location
    w_norm = ob.matrix_world.to_3x3() @ hit_normal
    return np.array(w_hit), np.array(w_norm)


# ===== GEOMETRY ===== #
def sort_grid_xy(co):
    """Sorts the points of a grid first on X then on Y"""
    sorted = co[np.argsort(co[:,0], kind='mergesort')]
    sorted = sorted[np.argsort(sorted[:,1], kind='mergesort')]
    return sorted


def add_randomness(arr, size=0.00000000001):
    """Assumes Nx3 shape and add a smal random value"""
    arr += np.random.rand(arr.shape[0], 3) * size


def manage_modifiers(ob, modes=None):
    if modes:
        for e, m in enumerate(ob.modifiers):
            m.show_viewport = modes[e]
        return
    
    modes = {}
    for e, m in enumerate(ob.modifiers):
        modes[e] = m.show_viewport
        m.show_viewport = False
    return modes        
    

# ===== GEOMETRY ===== #
def get_co_with_shape_keys(ob):
    """Gets the current coordinates including shape key
    effects but excluding modifier effects."""
    original_modifier_states = {}
    for mod in ob.modifiers:
        original_modifier_states[mod.name] = mod.show_viewport
        mod.show_viewport = False  # Disable the modifier
    prox = prox_object(ob)
    co = np.empty((len(prox.data.vertices), 3), dtype=np.float32)
    prox.data.vertices.foreach_get('co', co.ravel())

    for mod in ob.modifiers:
        mod.show_viewport = original_modifier_states[mod.name]
    return co


# ===== GEOMETRY ===== #
def prox_object(ob):
    """Returns object including modifier and
    shape key effects."""
    dg = bpy.context.evaluated_depsgraph_get()
    return ob.evaluated_get(dg)


def get_evaluated_mesh(ob, dg=None, apply_modifiers=False):
    """Returns a temporary mesh including modifier and shape key effects.
    Use this in place of accessing .data on an evaluated object."""
    #if dg is None:
        #dg = bpy.context.evaluated_depsgraph_get()
    # apply_modifiers=True ensures shape keys and modifiers are baked in.
    # preserve_all_data_layers=True if you need extras like UVs/vertex groups.
    return ob.to_mesh(preserve_all_data_layers=True)


# ===== GEOMETRY ===== #
def absolute_co(ob, ar=None, world=True, prox=None, abs_only=False):
    """Get proxy vert coords in world space.
    !!! Also returns prox object !!!
    Might need view update before matrix."""

    if prox is None:    
        prox = prox_object(ob)

    if ar is None:
        count = len(prox.data.vertices)
        ar = np.zeros((count, 3), dtype=D_TYPE_F)
    prox.data.vertices.foreach_get("co", ar.ravel())
    
    if abs_only:
        return apply_transforms(ob, ar)
    
    if world:    
        return apply_transforms(ob, ar), prox

    return ar


# ===== GEOMETRY ===== #
def co_to_shape(ob, co=None, key="current"):
    if co is None:
        co = get_co(ob)
    ob.data.shape_keys.key_blocks[key].data.foreach_set('co', co.ravel())
    ob.data.update()


def copy_shape_coords(ob, shape1, shape2):
    
    if ob.data.is_editmode:
        ob.update_from_editmode()
    
    k1 = ob.data.shape_keys.key_blocks[shape1]
    k2 = ob.data.shape_keys.key_blocks[shape2]
    co = np.empty((len(k1.data), 3), dtype=np.float32)
    k1.data.foreach_get('co', co.ravel())
    k2.data.foreach_set('co', co.ravel())
    

# ===== GEOMETRY ===== #
def get_shape_co_mode(ob=None, co=None, key='current'):
    """Edit or object mode"""

    if ob.data.is_editmode:
        ob.update_from_editmode()

    if co is None:
        co = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
        
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', co.ravel())
    return co


# ===== GEOMETRY ===== #
def get_shape_co(ob, shape='Key 1', co=None):
    if co is None:
        co = np.empty((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    ob.data.shape_keys.key_blocks[shape].data.foreach_get('co', co.ravel())
    return co


# ===== GEOMETRY ===== #
def set_shape_co(ob, shape, co):
    ob.data.shape_keys.key_blocks[shape].data.foreach_set('co', co.ravel())


# ===== GEOMETRY ===== #
def get_bmesh_co(obm, co=None):
    """To get the grabbed location of a vertex we
    need to get the data from the bmesh coords."""
    if co is None:
        return np.array([v.co for v in cloth.obm.verts], dtype=np.float32)

    for i in range(co.shape[0]):
        co[i] = obm.verts[i].co


# ===== GEOMETRY ===== #
def get_co_edit(ob, ar=None, key='current'):
    ob.update_from_editmode()
    if ar is None:
        c = len(ob.data.vertices)
        ar = np.empty((c, 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# ===== GEOMETRY ===== #
def get_proxy_co(ob, co=None, proxy=None, return_proxy=False):
    """Gets co with modifiers like cloth"""
    if proxy is None:

        dg = bpy.context.evaluated_depsgraph_get()
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    if return_proxy:
        return co, proxy, prox

    ob.to_mesh_clear()
    return co


# ===== GEOMETRY ===== #
def get_co(ob):
    """Returns Nx3 cooridnate set as numpy array"""
    co = np.empty((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    ob.data.vertices.foreach_get("co", co.ravel())
    return co


def get_mesh_co(mesh):
    """Returns Nx3 cooridnate set as numpy array"""
    co = np.empty((len(mesh.vertices), 3), dtype=D_TYPE_F)
    mesh.vertices.foreach_get("co", co.ravel())
    return co


# ===== GEOMETRY ===== #
def get_edge_centers(co, eidx, scale=0.5):
    """Start and end of edge
    has to match head and tail
    of bone if center of mass
    is not at 0.5."""
    head = co[eidx[:, 0]]
    vec = (co[eidx[:,1]] - head) * scale
    return head + vec


# ===== GEOMETRY ===== #
def slide_points_to_planes(e1, e2, origins, normals, intersect=False):
    '''Takes the start and end of an edge.
    Returns where it intersects the planes with a bool array for the
    edges that pass through the plane'''
    e_vecs = e2 - e1
    e1ors = origins - e1
    edge_dots = np.einsum('ij,ij->i', e_vecs, normals)
    dots = np.einsum('ij,ij->i', normals, e1ors)
    scale = np.nan_to_num(dots / edge_dots)    
    drop = + (e1 + e_vecs * scale[:, None])
    if intersect:
        inter = (scale >= 0) & (scale <= 1)
        return drop, inter, scale
    return drop


# ===== GEOMETRY ===== #
def inside_triangles(tris, points, margin=0.0):  # , cross_vecs):
    """Checks if points are inside triangles"""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, None]

    v2 = points - origins

    # ---------
    v0 = cross_vecs[:, 0]
    v1 = cross_vecs[:, 1]

    d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = d00_d11[:, 0]
    d11 = d00_d11[:, 1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    w = 1 - (u + v)
    # !!!! needs some thought
    # margin = 0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)
    return check, weights


# ===== GEOMETRY ===== #
def weight_plot(tco, weights):
    """For reploting weights from tris"""
    weight_plot = tco * weights[:, :, None]
    tri_plot = np.sum(weight_plot, axis=1)    
    return tri_plot


# ===== GEOMETRY ===== #
def get_edge_bounds(edges):
    """Takes an NxNx3 set of coords and returns
    the min and max for each edge/tri"""
    min = np.min(edges, axis=1)
    max = np.max(edges, axis=1)
    return [min, max]


# ===== GEOMETRY ===== #
def triangle_swept_bounds(tris):
    """
    tris: (N, 6, 3) array of moving triangles
          0-2 = vertices at t
          3-5 = vertices at t+dt
    Returns:
        bmin, bmax arrays of shape (N,3)
    """
    bmin = tris.min(axis=1)  # (N,3)
    bmax = tris.max(axis=1)  # (N,3)
    return bmin, bmax


def get_bounds(co):
    """Returns the min and max corner
    of the coordinates"""
    bmin = np.min(co, axis=0)
    bmax = np.max(co, axis=0)
    return bmin, bmax
    
    
def box_from_bounds(bmin, bmax, name="Box"):

    # Unpack the min and max coordinates
    x_min, y_min, z_min = bmin
    x_max, y_max, z_max = bmax

    # Define the 8 corners of the box
    co = [
        (x_min, y_min, z_min),  # Vertex 0: Bottom-front-left
        (x_max, y_min, z_min),  # Vertex 1: Bottom-front-right
        (x_max, y_max, z_min),  # Vertex 2: Bottom-back-right
        (x_min, y_max, z_min),  # Vertex 3: Bottom-back-left
        (x_min, y_min, z_max),  # Vertex 4: Top-front-left
        (x_max, y_min, z_max),  # Vertex 5: Top-front-right
        (x_max, y_max, z_max),  # Vertex 6: Top-back-right
        (x_min, y_max, z_max),  # Vertex 7: Top-back-left
    ]

    # Define the 12 edges of the box
    edges = [
        # Bottom face edges
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face edges
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges connecting bottom and top faces
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    if name in bpy.data.objects:
        ob = bpy.data.objects[name]
        ob.data.vertices.foreach_set('co', np.array(co).ravel())
        ob.data.update()
        return
        
    ob_from_py_data(co, faces=[], edges=edges, name=name)


def box_around_selected():
    sel = bpy.context.selected_objects
    obs = [ob for ob in sel if ob.type == "MESH"]
    if len(obs) == 0:
        popup_error("No selected meshes")
        return
    cos = [absolute_co(ob, abs_only=True) for ob in obs]
    bounds = [get_bounds(co) for co in cos]
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    bmin = np.min(mins, axis=0)
    bmax = np.max(maxs, axis=0)
    box_from_bounds(bmin, bmax, name="Box")
        

def delete_by_name(name="PLEASE DON'T KILL ME!'"):
    """Delete object and object data if its name
    starts with arg name."""
    for ob in bpy.data.objects:
        if ob.name.startswith(name):
            mesh = ob.data
            bpy.data.objects.remove(ob)
            if mesh:    
                bpy.data.meshes.remove(mesh)


def select_by_name(name="PICK ME!'"):
    """Select object if its name
    starts with arg name."""
    for ob in bpy.data.objects:
        ob.select_set(False)
        if ob.name.startswith(name):
            ob.select_set(True)


def test_box():
    ob = bpy.context.object
    bmin, bmax = get_bounds(get_co(ob))
    box_from_bounds(bmin, bmax, name="Box")
    e1 = bpy.data.objects['e1']    
    e2 = bpy.data.objects['e2']    
    e1.location = bmin
    e2.location = bmax


# ===== GEOMETRY ===== #
def get_tri_normals(tco, normalize=True):
    """Creates a tangent universe consisting entirely
    of two bowls of cereal having a conversation
    about hip replacement surgery."""
    tv1 = tco[:, 1] - tco[:, 0]
    tv2 = tco[:, 2] - tco[:, 0]
    #cross = np.cross(tv1, tv2)
    cross = fastest_cross_product(tv1, tv2)
    if normalize:
        return u_vecs(cross)    
    return cross
    

# ===== GEOMETRY ===== #
def get_vertex_normals(ob, mesh=False):
    """Vertex normals from object using blender
    calculated normals."""
    if mesh:
        normals = np.zeros((len(ob.vertices), 3), dtype=D_TYPE_F)
        ob.vertices.foreach_get('normal', normals.ravel())
        return normals
    normals = np.zeros((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    ob.data.vertices.foreach_get('normal', normals.ravel())
    return normals


# ===== GEOMETRY ===== #
def get_poly_normals(ob):
    normals = np.empty((len(ob.data.polygons), 3), dtype=D_TYPE_F)
    ob.data.polygons.foreach_get('normal', normals.ravel())
    return normals
    

# ===== GEOMETRY ===== #
def get_poly_normals_mesh(mesh):
    normals = np.empty((len(mesh.polygons), 3), dtype=D_TYPE_F)
    mesh.polygons.foreach_get('normal', normals.ravel())
    return normals


# ===== GEOMETRY ===== #
def xyz_world(ob, normalize=False):
    """Get the xyz axis vectors of an object.
    Assumes a scale of 1, 1, 1"""
    xyz = np.array(ob.matrix_world, dtype=D_TYPE_F)[:3, :3].T
    if normalize:
        xyz /= np.array(ob.scale, dtype=np.float32)[:, None]
    return xyz


# ===== GEOMETRY ===== #
def apply_transforms(ob, co):
    """Get co in world space.
    Respect: location, rotation, scale."""
    m = np.array(ob.matrix_world, dtype=D_TYPE_F)
    mat = m[:3, :3].T
    return co @ mat + m[:3, 3]


def revert_transforms(ob, co):
    """World coordinates applied to local space.

    For M = R*S (rotation times axis scale) the inverse is (M / |col|^2).T.
    Coordinates here are ROW vectors, so local = (world - t) @ inverse(M).T,
    which is plain M / |col|^2.  An extra .T here once applied the inverse
    rotation twice: harmless on an unrotated cloth, but a cloth object with a
    rotation saw every collider turned the wrong way and fell through it."""
    m = np.array(ob.matrix_world, dtype=D_TYPE_F)
    r = m[:3, :3]
    return (co - m[:3, 3]) @ (r / np.einsum('ij,ij->j', r, r))


# ===== GEOMETRY ===== #
def revert_transforms_(ob, co, update_view=False):
    """World coordinates applied
    to local space."""
    m = np.array(ob.matrix_world, dtype=D_TYPE_F)
    npscale = np.array(ob.scale, dtype=D_TYPE_F)
    mat = m[:3, :3] / (npscale ** 2)
    return (co - m[:3, 3]) @ mat
        

# ===== GEOMETRY ===== #
def apply_rotation(ob, co, update_view=False):
    """When applying vectors such as normals
    we only need rotation.
    Might need view update before matrix.
    !!! Forces will be scaled by object scale !!!"""
    m = np.array(ob.matrix_world, dtype=D_TYPE_F)
    mat = m[:3, :3].T
    return co @ mat


# ===== GEOMETRY ===== #
def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate. Forces need to be scaled.
    Might need view update before matrix."""
    m = np.array(ob.matrix_world, dtype=D_TYPE_F)
    npscale = np.array(ob.scale, dtype=D_TYPE_F)
    mat = m[:3, :3] / (npscale ** 2)
    return co @ mat


# ===== GEOMETRY ===== #
def co_to_matrix_space(co, ob1, ob2):
    """Finds the world coordinates of ob1
    and puts them in ob2 coordinate space"""
        
    local_matrix = np.linalg.inv(ob2.matrix_world) @ np.array(ob1.matrix_world, dtype=np.float32)
    local_co = co @ local_matrix[:3, :3].T
    local_co += local_matrix[:3, 3]
    return local_co


# ===== GEOMETRY ===== #
def u_vecs(vecs, dist=False, out=None):
    length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs, out=out))
    u_v = vecs / length[:, None]
    if dist:
        return u_v, length
    return u_v


def u_vecs_out(vecs, dist=False, out=None, length_out=None):
    length = np.sqrt(np.einsum('ij,ij->i', vecs, vecs, out=length_out))
    if out is not None:
        np.divide(vecs, length[:, None], out=out)
        u_v = out
    else:
        u_v = vecs / length[:, None]
    if dist:
        return u_v, length
    return u_v


# ===== GEOMETRY ===== #
def closest_points_edges(vecs, origins, p):
    '''Returns the location of the points on the edges'''
    vec2 = p - origins
    d = np.einsum('ij,ij->i', vecs, vec2) / np.einsum('ij,ij->i', vecs, vecs)
    cp = origins + vecs * d[:, None]
    return cp, d


# ===== GEOMETRY ===== #
def closest_point_edge(vec, origin, p, clip=False):
    '''Returns the location of the points on the edges'''
    vec2 = p - origin
    d = (vec @ vec2) / (vec @ vec)
    if clip:
        d = np.clip(d, 0.0, 1.0)
    cp = origin + vec * d
    return cp, d


# ===== GEOMETRY ===== #
def closest_point_edges(vecs, origins, p):
    '''Returns the location of the point on the edges'''
    vec2 = p - origins
    d = np.einsum('ij,ij->i', vecs, vec2) / np.einsum('ij,ij->i', vecs, vecs)
    cp = origins + vecs * d[:, None]
    return cp, d


# ===== GEOMETRY ===== #
def closest_points_on_tri_edges_readable(v_curr, v_next, edge_vecs, v2):
    """
    Computes the closest point on each of the three edges of the triangle to the query point.
    
    For each triangle-point pair, returns a (N, 3, D) array of closest points, one per edge:
    - Edge 0: from vertex 0 to vertex 1
    - Edge 1: from vertex 1 to vertex 2
    - Edge 2: from vertex 2 to vertex 0
    
    If the closest point is outside the edge, returns the appropriate edge vertex.
    
    Args:
        v_curr: Current vertices (N, 3, D)
        v_next: Next vertices (N, 3, D) - rolled version of v_curr
        edge_vecs: Edge vectors (N, 3, D) - v_next - v_curr
        v2: Query points relative to first vertex (N, D)
    
    Returns:
        closest: Closest points on each edge (N, 3, D)
    """
    N, _, D = v_curr.shape
    closest = np.zeros((N, 3, D))
    
    # Compute edge lengths squared for all edges
    edge_lens_sq = np.einsum('ijk,ijk->ij', edge_vecs, edge_vecs)  # (N, 3)
    
    # For each edge, compute the projection parameter
    # Edge 0: from vertex 0 to vertex 1
    # Vector from vertex 0 to query point is v2
    dot_0 = np.einsum('ij,ij->i', edge_vecs[:, 0], v2)  # (N,)
    t0 = np.divide(dot_0, edge_lens_sq[:, 0], out=np.zeros_like(dot_0), where=edge_lens_sq[:, 0] != 0)
    t0_clamped = np.clip(t0, 0.0, 1.0)
    closest[:, 0] = v_curr[:, 0] + t0_clamped[:, None] * edge_vecs[:, 0]
    
    # Edge 1: from vertex 1 to vertex 2
    # Vector from vertex 1 to query point
    v_from_1 = v2 - edge_vecs[:, 0]  # query point relative to vertex 1
    dot_1 = np.einsum('ij,ij->i', edge_vecs[:, 1], v_from_1)  # (N,)
    t1 = np.divide(dot_1, edge_lens_sq[:, 1], out=np.zeros_like(dot_1), where=edge_lens_sq[:, 1] != 0)
    t1_clamped = np.clip(t1, 0.0, 1.0)
    closest[:, 1] = v_curr[:, 1] + t1_clamped[:, None] * edge_vecs[:, 1]
    
    # Edge 2: from vertex 2 to vertex 0
    # Vector from vertex 2 to query point
    v_from_2 = v2 - edge_vecs[:, 0] - edge_vecs[:, 1]  # query point relative to vertex 2
    dot_2 = np.einsum('ij,ij->i', edge_vecs[:, 2], v_from_2)  # (N,)
    t2 = np.divide(dot_2, edge_lens_sq[:, 2], out=np.zeros_like(dot_2), where=edge_lens_sq[:, 2] != 0)
    t2_clamped = np.clip(t2, 0.0, 1.0)
    closest[:, 2] = v_curr[:, 2] + t2_clamped[:, None] * edge_vecs[:, 2]
    
    return closest


# ===== GEOMETRY ===== #
def closest_points_on_tri_edges_fast(c_verts, v_curr, v_next, edge_vecs):
    """
    Computes the closest point on each of the three edges of the triangle to the query point.
    If the closest point is outside the edge, returns the appropriate edge vertex.
    
    Args:
        v_curr: Current vertices (N, 3, D)
        v_next: Next vertices (N, 3, D) - rolled version of v_curr
        edge_vecs: Edge vectors (N, 3, D) - v_next - v_curr
        v2: Query points relative to first vertex (N, D)
    
    Returns:
        closest: Closest points on each edge (N, 3, D)
    """

    edge_lens_sq = np.einsum('ijk,ijk->ij', edge_vecs, edge_vecs)  # (N, 3)
    v_ori_vecs = c_verts[:, None, :] - v_curr
    dots = np.einsum('ijk,ijk->ij', edge_vecs, v_ori_vecs)  # (N, 3)    
    t = np.divide(dots, edge_lens_sq, out=np.zeros_like(dots), where=edge_lens_sq != 0)
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = v_curr + t_clamped[:, :, None] * edge_vecs  # (N, 3, D)
    
    return closest, v_ori_vecs


# ===== GEOMETRY ===== #
def closest_face_normal(co, t_data=None):
    """Manages v f and e types to get
    normal from np_closest_point_mesh()"""
    location, type, index = np_closest_point_mesh(co, t_data=t_data)
    obm_ex = t_data["tobm"]
    
    if type == 'f':
        return np.array(obm_ex.faces[index].normal, dtype=D_TYPE_F)
    if type == 'v':
        return np.array(obm_ex.verts[index].normal, dtype=D_TYPE_F)
    if type == 'e':
        eco = np.array([f.normal for f in obm_ex.edges[index].link_faces], dtype=D_TYPE_F)
        return np.mean(eco, axis=0)


# ===== GEOMETRY ===== #
def point_to_triangle_forces(t_weights, balance=False):
    # t_weights is Nx3, each row sums to 1.0
    # Correction scale: 1 / sum of squared weights
    #scale = 1.0 / np.einsum('ij,ij->i', t_weights, t_weights)
    scale = 1 / np.max(t_weights, axis=1)
    
    if balance:
        multipliers = t_weights * scale[:, None] * 0.5
    else:
        multipliers = t_weights * scale[:, None]
    
    return multipliers


# ===== GEOMETRY ===== #
def set_magnitude(v, m):
    """Takes Nx3 vectors and set the magnitude to m"""
    dist = np.linalg.norm(v, axis=1, keepdims=True)
    scale = m / dist
    return v * scale, dist.ravel()


# ===== GEOMETRY ===== #
def edge_to_edge_forces(e1, e2, e3, e4, target_dist, balance=False, enorms=None, under=None, flip=None):
    cp1, cp2, s, t = edges_to_edges_clip(e1, e2, e3, e4)
    
    delta = cp1 - cp2
    dist = np.linalg.norm(delta, axis=1)
    
    in_range = np.abs(dist) < target_dist
    if flip is not None:
        #in_range[flip] = True
        pass
    
    
    if False:    
        e_comp = compare_vecs(enorms, delta)
        in_range[e_comp < 0] = False
    
    delta = delta[in_range]    
    dist = dist[in_range]
    s = s[in_range]     
    t = t[in_range]    
    
    normal = delta / dist[:, None]
    penetration = (target_dist - dist)
    
    if False:
        under = under[in_range]
        back = (target_dist + dist[under]) * -1.0
        penetration[under] = back
    
    if False:    
        if enorms is not None:    
            e_comp = compare_vecs(enorms[in_range], delta)
            under = e_comp < 0.0
            
            penetration[under] = dist[under] + target_dist
            normal[under] *= -1.0

    scale1 = 1.0 / ((1 - s)**2 + s**2)
    
    # Each edge is responsible for half the total correction
    if balance:    
        scale2 = 1.0 / ((1 - t)**2 + t**2)

        w1 = (1 - s) * scale1
        w2 = s * scale1
        w3 = (1 - t) * scale2
        w4 = t * scale2

        half_penetration = penetration * 0.5
        f1 = w1 * half_penetration  # e1 weight
        f2 = w2 * half_penetration  # e2 weight
        f3 = w3 * half_penetration  # e3 weight
        f4 = w4 * half_penetration  # e4 weight

        fe1 =  f1[:, None] * normal
        fe2 =  f2[:, None] * normal
        fe3 = -f3[:, None] * normal
        fe4 = -f4[:, None] * normal
        
        return w1, w2, w3, w4, fe1, fe2, fe3, fe4, in_range
    w1 = (1 - s) * scale1
    w2 = s * scale1
    f1 = w1 * penetration  # e1 weight
    f2 = w2 * penetration  # e2 weight
    
    # Edge 1 pushes along +normal, edge 2 pushes along -normal
    force_e1 =  f1[:, None] * normal
    force_e2 =  f2[:, None] * normal
    
    return force_e1, force_e2, in_range


# ===== GEOMETRY ===== #
def np_closest_point_mesh(co, ob=None, t_data=None, return_weights=False):
    """Returns closest point on mesh
    and type v, f, or e depending
    on if the closest place was
    on a vert, a face, or an edge."""

    face = False
    edge = False
    edge_weight = None
    tri_weight = None

    # faces
    if t_data is None:
        tridex, teidx, tobm = get_tridex_with_edges(ob, teidx=True, tmesh=False)
    else:
        tridex, teidx, tobm = t_data['tridex'], t_data['teidx'], t_data['tobm']

    if ob is None:
        obco = np.array([v.co for v in tobm.verts], dtype=D_TYPE_F)
    else:
        obco = get_co(ob)

    tris = obco[tridex]
    check, weights = inside_triangles(tris, co, margin=0.0)
    checked_weights = weights[check]
    
    if np.any(check):
        face = True
        tidx = np.arange(tris.shape[0])[check]
        weight_plot = tris[check] * weights[check][:, :, None]
        locs = np.sum(weight_plot, axis=1)
        dif = co - locs
        dots = np.einsum('ij,ij->i', dif, dif)
        f_min = np.argmin(dots)
        f_loc = locs[f_min]
        tid = tidx[f_min]
        tri_weight = checked_weights[f_min]
        
    # edges
    teco = obco[teidx]
    vecs = teco[:, 1] - teco[:, 0]
    cpoes, dots = closest_point_edges(vecs, teco[:, 0], co)
    check = (dots > 0.0) & (dots < 1.0)
    
    if np.any(check):
        edge = True
        check_dots = dots[check]
        eidc = np.arange(check.shape[0])[check]
        checked = cpoes[check]
        dif = co - checked
        dots = np.einsum('ij,ij->i', dif, dif)
        emin = np.argmin(dots)
        e_loc = checked[emin]
        eid = eidc[emin]
        edge_weight = check_dots[emin]

    # verts
    dif = co - obco
    dots = np.einsum('ij,ij->i', dif, dif)
    vmin = np.argmin(dots)
    v_loc = obco[vmin]

    trifecta = [v_loc]
    type = ['v']
    ind = [vmin]
    tobm.verts.ensure_lookup_table()
    tobm.verts[vmin].normal

    if face:
        trifecta += [f_loc]
        type += ['f']
        ind += [tid]

    if edge:
        trifecta += [e_loc]
        type += ['e']
        ind += [eid]

    npl = np.array(trifecta, dtype=D_TYPE_F)
    dif = co - npl
    dots = np.einsum('ij,ij->i', dif, dif)
    amin = np.argmin(dots)
    loc = npl[amin]  # and all the chistians said "amin."
    t = type[amin]
    index = np.array(ind, dtype=D_TYPE_I)[amin]
    # location, type, index
    if return_weights:
        return loc, t, index, tri_weight, tridex, teidx, edge_weight
    return loc, t, index


# ===== GEOMETRY ===== #
def vec_to_vec_pivot(ph, edges, vecs, rot_edges, factor=0.5, target_length=None, x_edges=False):
    """Rotate edges to match vecs respecting
    a pivot defined by a factor that plots
    the pivot along the edge. The factor
    should be defined by the bone center
    of mass.
    !!! Assumes vecs are normalized !!!"""

    rot_edges.shape = ph.rel_rot_shape
    if x_edges:
        half = vecs * factor
        rot_edges[:, 0] = ph.relative_edge_mids - half
        rot_edges[:, 1] = ph.relative_edge_mids + half
        return rot_edges
        
    np.mean(edges, axis=1, out=ph.relative_edge_mids)

    if target_length is not None:
        dist = target_length * 0.5

    rot_edges[:, 0] = ph.relative_edge_mids - (vecs * dist)
    rot_edges[:, 1] = ph.relative_edge_mids + (vecs * dist)
    return rot_edges


# ===== GEOMETRY ===== #
def quaternion_to_euler(w, x, y, z):
    """
    Convert a quaternion (w, x, y, z) into Euler angles (roll, pitch, yaw).
    
    :param w: Scalar component of quaternion
    :param x: X component of quaternion
    :param y: Y component of quaternion
    :param z: Z component of quaternion
    :return: Tuple of Euler angles (roll, pitch, yaw) in radians
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    c = 180 / np.pi

    return c * roll - 180, c * pitch, c * yaw


# ===== GEOMETRY ===== #
def rotate_vector(vector, axis, angle, origin=(0, 0, 0)):
    """
    Rotate a 3D vector around an arbitrary axis.
    
    Parameters:
    vector (array-like): The vector to be rotated.
    axis (array-like): The axis of rotation.
    angle (float): The angle of rotation in radians.
    origin (array-like, optional): The pivot point. Defaults to (0, 0, 0).
    
    Returns:
    numpy.ndarray: The rotated vector.
    """
    # Convert inputs to numpy arrays
    vector = np.asarray(vector)
    axis = np.asarray(axis)
    origin = np.asarray(origin)
    
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    
    # Translate vector to origin
    vector = vector - origin
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = (
        np.eye(3) + 
        np.sin(angle) * K + 
        (1 - np.cos(angle)) * np.dot(K, K)
    )
    
    # Apply the rotation
    rotated_vector = np.dot(rotation_matrix, vector)
    
    # Translate back
    rotated_vector = rotated_vector + origin
    
    return rotated_vector


def rotate_points(points, axis, angle, origin=None):
    "Rodrigues rotate points"
    # Translate points to origin
    if origin is not None:    
        points = points - origin
        
    K = np.array([
        [0, axis[2], -axis[1]],
        [-axis[2], 0, axis[0]],
        [axis[1], -axis[0], 0]
    ])
    rotation_matrix_T = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    rotated_points = points @ rotation_matrix_T  # No transpose needed        
    
    # Translate back
    if origin is not None:    
        rotated_points += origin
    
    return rotated_points


def get_rodrigues_matrix(axis, angle, normalize=False, conventional=False):
    """Extract just the 3x3 rotation matrix from Rodrigues' formula.
    If conventional is False build it already transposed."""
    
    if normalize:
        axis /= np.linalg.norm(axis)  # ensure unit vector
    
    if conventional:
        K = np.array([
            [0, -axis[2],  axis[1]],
            [axis[2],  0, -axis[0]],
            [-axis[1],  axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        K = np.array([
            [0,  axis[2], -axis[1]],
            [-axis[2], 0,  axis[0]],
            [axis[1], -axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _skew(v):
    """Skew-symmetric (cross-product) matrix of vector v."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


def orthonormalize_matrix_svd(m):
    """Re-orthonormalize the rotation component of a 4x4 or 3x3 numpy matrix via SVD"""
    U_svd, _, Vt = np.linalg.svd(m[:3, :3])
    m[:3, :3] = U_svd @ Vt
    return m


def orthonormalize_matrix(m):
    """Re-orthonormalize the rotation component of a 4x4 or 3x3 numpy matrix via Gram-Schmidt"""
    x = m[:3, 0]
    y = m[:3, 1]
    x = x / np.sqrt(x @ x)
    y = y - (y @ x) * x
    y = y / np.sqrt(y @ y)
    z = np.cross(x, y)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    return m


def rotation_matrix_from_vectors(a, b, normalize=False, conventional=False):
    if normalize:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = a @ b

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):
        perp = np.array([1, 0, 0]) if not np.isclose(abs(a[0]), 1.0) else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        if conventional:
            K = _skew(v)
        else:
            K = _skew(-v)  # flipped skew for transposed convention
        return np.eye(3) + 2 * (K @ K)

    if conventional:
        K = _skew(v)
        return np.eye(3) + K + (K @ K) * (1 / (1 + c))
    else:
        K = _skew(-v)  # flipped skew = pre-transposed result
        return np.eye(3) + K + (K @ K) * (1 / (1 + c))


def get_face_data(ob):
    """Gets face centers and normals disabling modifiers
    Fast as my neighbor's dog when I'm chasing it down with a hammer."""
    modes = manage_modifiers(ob)
    prox = prox_object(ob)        
    temp_mesh = bpy.data.meshes.new_from_object(prox)
    face_centers = get_face_centers_mesh(temp_mesh)
    face_normals = get_poly_normals_mesh(temp_mesh)
    manage_modifiers(ob, modes)
    bpy.data.meshes.remove(temp_mesh)
    return face_centers, face_normals


def rotation_matrices(axes, angles):
    N = len(angles)
    
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = axes[:, 2]
    K[:, 0, 2] = -axes[:, 1]
    K[:, 1, 0] = -axes[:, 2]
    K[:, 1, 2] = axes[:, 0]
    K[:, 2, 0] = axes[:, 1]
    K[:, 2, 1] = -axes[:, 0]
    
    K2 = K @ K
    
    sin_angles = np.sin(angles)[:, None, None]
    cos_angles = np.cos(angles)[:, None, None]
    
    rotation_mats = np.eye(3)[None, :, :] + sin_angles * K + (1 - cos_angles) * K2
    
    return rotation_mats



# ===== GEOMETRY ===== #
def remove_doubles(group, threshold=.0001):
    """finds coincident points and returns a bool array eliminating all but the first
    occurance of the coincident points"""
    x = group - group[:, None]
    dist = np.einsum('ijk, ijk->ij', x, x)
    pairs = dist < threshold
    doubles = np.sum(pairs, axis=0) > 1
    idx = np.arange(len(group))[doubles]
    all_true = np.ones(len(group), dtype=np.bool)
    for i in idx:
        this = np.all((group[i] - group[idx]) == 0, axis=1)
        all_true[idx[this][1:]] = False
    return all_true


# ===== GEOMETRY ===== #
def coincident_points(group_a, group_b, threshold=.0001, inverse=True):
    """finds the index of points in group a that match the location of at
    least one point in group b. Returns the inverse by default: points that have no match
    returns a bool array matching the first dimension of group_a"""
    x = group_b - group_a[:, None]
    dist = np.einsum('ijk, ijk->ij', x, x)
    min_dist = np.min(dist, axis=1)
    if inverse:
        return min_dist > threshold
    return min_dist < threshold


# ===== GEOMETRY ===== #
def rotate_vectors(vectors, axis, angle):
    """
    Rotate an array of vectors around a given axis by a specified angle.
    
    Parameters:
    vectors (numpy.ndarray): An Nx3 array of vectors to be rotated.
    axis (numpy.ndarray): A 3D vector representing the axis of rotation.
    angle (float): The angle of rotation in radians.
    
    Returns:
    numpy.ndarray: An Nx3 array of rotated vectors.
    """
    # Ensure inputs are numpy arrays
    vectors = np.asarray(vectors)
    axis = np.asarray(axis)
    
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    
    # Compute rotation matrix components
    cosa = np.cos(angle)
    sina = np.sin(angle)
    vera = 1 - cosa
    x, y, z = axis
    
    # Compute rotation matrix
    rot_matrix = np.array([
        [cosa + x*x*vera,    x*y*vera - z*sina, x*z*vera + y*sina],
        [y*x*vera + z*sina,  cosa + y*y*vera,   y*z*vera - x*sina],
        [z*x*vera - y*sina,  z*y*vera + x*sina, cosa + z*z*vera]
    ])
    
    # Apply rotation to all vectors
    return np.dot(vectors, rot_matrix.T)


# ===== GEOMETRY ===== #
def compare_direction(v1, v2):
    """Returns a bool array indicating
    vecs pointing the same direction
    (their dot products are zero or
    positive)"""
    dots = np.einsum('ij,ij->i', v1, v2)
    return dots >= 0.0


# ===== GEOMETRY ===== #
def measure_vecs_slower(vecs): # was once faster a long time ago in a galaxy far far away
    return np.sqrt(np.einsum('ij,ij->i', vecs, vecs))

#def measure_vecs(vecs):
#    return np.linalg.norm(vecs, axis=1)

# ===== GEOMETRY ===== #
def measure_vecs(vecs, out=None, out2=None):
    return np.sqrt(np.einsum('ij,ij->i', vecs, vecs, out=out), out=out2)

# ===== GEOMETRY ===== #
def compare_vecs(v1, v2):
    return np.einsum('ij,ij->i', v1, v2)


# ===== GEOMETRY ===== #
def fastest_cross_product(a, b, c=None):
    """Faster than np.cross for NX3 to NX3.
    Can write to "c" if an array is provided
    for extra speed."""
    if c is None:    
        c = np.empty_like(a)
    c[:, 0], c[:, 1], c[:, 2] = (
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    )
    return c


def fastest_cross_product_n_3(a, b, c=None):
    """Faster than np.cross for NX3 to 3.
    Can write to "c" if an array is provided
    for extra speed."""
    if c is None:    
        c = np.empty_like(a)
    
    # b is now just a length-3 vector, so we index it directly
    c[:, 0] = a[:, 1] * b[2] - a[:, 2] * b[1]
    c[:, 1] = a[:, 2] * b[0] - a[:, 0] * b[2]
    c[:, 2] = a[:, 0] * b[1] - a[:, 1] * b[0]
    
    return c


def fastest_cross_product_single(a, b, c=None):
    """Faster than np.cross for 3 to 3.
    Can write to "c" if an array is provided
    for extra speed."""
    if c is None:    
        c = np.empty_like(a)
    c[0], c[1], c[2] = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )
    return c


# ===== GEOMETRY ===== #
def edge_to_edge(e1, e2, e3, e4):
    """Single edge to edge segment."""
    v1 = e2 - e1
    v2 = e4 - e3
    v3 = e3 - e1

    cross = np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ])

    cross_dot = cross @ cross
    d = (v3 @ cross) / cross_dot
    spit = cross * d
    cp1 = e1 + spit
    vec2 = cp1 - e3
    d2 = (vec2 @ v2) / (v2 @ v2)
    nor = v2 * d2
    cp2 = e3 + nor
    normal = cp1 - cp2
    or_vec = e1 - cp2
    e_dot = normal @ v1
    e_n_dot = normal @ or_vec
    scale = e_n_dot / e_dot
    p_on_p = (or_vec - v1 * scale) + cp2
    return p_on_p, p_on_p + spit


def edge_to_edge_optimized(e1, e2, e3, e4):
    """Optimized single edge to edge segment.
    Faster than above function: edge_to_edge"""
    v1 = e2 - e1  # Direction of segment 1
    v2 = e4 - e3  # Direction of segment 2
    v3 = e1 - e3  # Vector between start points
    
    # Precompute dot products
    d1_d1 = v1 @ v1
    d1_d2 = v1 @ v2
    d2_d2 = v2 @ v2
    d1_v3 = v1 @ v3
    d2_v3 = v2 @ v3
    
    # Solve for parameters s and t
    denom = d1_d1 * d2_d2 - d1_d2 * d1_d2
    s = (d1_d2 * d2_v3 - d2_d2 * d1_v3) / denom
    t = (d1_d1 * d2_v3 - d1_d2 * d1_v3) / denom
    
    # Closest points
    cp1 = e1 + s * v1
    cp2 = e3 + t * v2
    
    return cp1, cp2


def edges_to_edges(e1, e2, e3, e4):
    """Vectorized for N segment pairs."""
    v1 = e2 - e1  # Shape: (N, 3)
    v2 = e4 - e3
    v3 = e1 - e3
    
    d1_d1 = np.einsum('ij,ij->i', v1, v1)
    d1_d2 = np.einsum('ij,ij->i', v1, v2)
    d2_d2 = np.einsum('ij,ij->i', v2, v2)
    d1_v3 = np.einsum('ij,ij->i', v1, v3)
    d2_v3 = np.einsum('ij,ij->i', v2, v3)
    
    denom = d1_d1 * d2_d2 - d1_d2 * d1_d2
    denom_inv = 1.0 / denom
    
    s = (d1_d2 * d2_v3 - d2_d2 * d1_v3) * denom_inv
    t = (d1_d1 * d2_v3 - d1_d2 * d1_v3) * denom_inv
    
    cp1 = e1 + s[:, None] * v1
    cp2 = e3 + t[:, None] * v2
    
    return cp1, cp2


def edges_to_edges_clip(e1, e2, e3, e4, clipping=True):
    v1 = e2 - e1
    v2 = e4 - e3
    v3 = e1 - e3

    d1_d1 = np.einsum('ij,ij->i', v1, v1)
    d1_d2 = np.einsum('ij,ij->i', v1, v2)
    d2_d2 = np.einsum('ij,ij->i', v2, v2)
    d1_v3 = np.einsum('ij,ij->i', v1, v3)
    d2_v3 = np.einsum('ij,ij->i', v2, v3)

    denom = d1_d1 * d2_d2 - d1_d2**2
    denom_inv = np.where(np.abs(denom) > 1e-12, 1.0 / denom, 0.0)

    s_num = d1_d2 * d2_v3 - d2_d2 * d1_v3
    t_num = d1_d1 * d2_v3 - d1_d2 * d1_v3

    s = s_num * denom_inv
    t = t_num * denom_inv

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    cp1 = e1 + s[:, None] * v1
    cp2 = e3 + t[:, None] * v2

    return cp1, cp2, s, t


test_me = True
test_me = False
if test_me:
    print("testing edge to edge")
    names = ['e1', 'e2','e3', 'e4','e5', 'e6']
    obs = [bpy.data.objects[e] for e in names]
    e1, e2, e3, e4, e5, e6, = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
    locs = np.array([e.location for e in obs])
    
    c1 = locs[0]
    c2 = locs[1]
    c3 = locs[2]
    c4 = locs[3]
    
    T = time.time()
    for i in range(1000):
        cp1, cp2 = edge_to_edge(c1, c2, c3, c4)
    print(time.time() - T, "t1")
    
    T = time.time()
    for i in range(1000):
        cp1, cp2 = edge_to_edge_optimized(c1, c2, c3, c4)
    print(time.time() - T, "t2")

    e5.location = cp1
    e6.location = cp2


def test_e():

    ob = bpy.context.object
    co = get_co(ob)
    
    T = time.time()
    for i in range(10000):    
        spit = edge_to_edge(co[0], co[1], co[2], co[3])
    print(time.time() - T)
    
    T = time.time()
    for i in range(10000):    
        spit = edge_to_edge2(co[0], co[1], co[2], co[3])
    print(time.time() - T)
    
    e1 = bpy.data.objects['ee1']
    e2 = bpy.data.objects['ee2']
    #e1.location = spit[0]
    #e2.location = spit[1]
    e1.location = spit[0]
    e2.location = spit[1]
    return    
    T = time.time()
    for i in range(10000):
        spit = shortest_orthogonal_edge(co[0], co[1], co[2], co[3])
    print(time.time() - T)

    T = time.time()
    for i in range(10000):    
        spit = edge_to_edge_optimized(co[0], co[1], co[2], co[3])
    print(time.time() - T)


# ===== GEOMETRY ===== #
def matrix_from_verts(verts):
    """Converts the coordinates from
    triangles into 3x3 matricies.
    Takes N sets of 3 coordinates,
    generates a 3x3 matrix using c1 - c0
    as X then the normal as Z.
    Y is the cross of X and the normal ("Obviously," he said with a condescending sigh)"""
    count = verts.shape[0]
    M = np.empty((count, 3, 3), dtype=np.float32)
    x = verts[:, 1] - verts[:, 0]
    vec2 = verts[:, 2] - verts[:, 0]
    cross = fastest_cross_product(x, vec2)
    normal = u_vecs(cross)
    ux = u_vecs(x)
    y = u_vecs(fastest_cross_product(normal, ux))

    M[:, 0] = ux
    M[:, 1] = y
    M[:, 2] = normal
    return M


# ===== GEOMETRY ===== #
def uni_pairs(eidx):
    """Remove duplicates and mirror duplicates."""    
    s_eidx = np.sort(eidx, axis=1)
    uni_eidx = np.unique(s_eidx, axis=0)
    return uni_eidx
    

# ===== GEOMETRY ===== #
def normal_smooth_groq(ob, iters=4, sub_iters=4, strength=1.0, co=None, tridex=None, eidx=None):
    """Uses blender vertex normals to tilt edges
    towards the mean effectively smoothing along
    the normal. Thank you Elon Musk for optimizing."""
    strength = strength * 0.23
    
    if co is None:
        co = get_co(ob)
    if eidx is None:    
        eidx = get_eidx(ob)
    up = uni_pairs(eidx)
    n_verts = len(co)
    sum_dot = np.zeros(n_verts)
    
    # Precompute vertex degrees
    degree = np.zeros(n_verts, dtype=np.float32)
    np.add.at(degree, up.ravel(), 1)
    if tridex is None:    
        tridex = get_tridex_3(ob)
    for it in range(iters):    
        norms = get_vertex_normals_np(ob, co, tridex)
        dot_c_v = np.einsum('ij,ij->i', co, norms)
        
        # For each edge, compute dot products
        dot_c1_n0 = np.einsum('ij,ij->i', co[up[:, 1]], norms[up[:, 0]])
        dot_c0_n1 = np.einsum('ij,ij->i', co[up[:, 0]], norms[up[:, 1]])
        
        # Sum neighbor contributions for each vertex
        sum_dot[:] = 0.0
        np.add.at(sum_dot, up[:, 0], dot_c1_n0)
        np.add.at(sum_dot, up[:, 1], dot_c0_n1)
        
        # Compute scalar movement
        scalar_move = (sum_dot - degree * dot_c_v) * strength

        # Apply movement along normals
        movement = scalar_move[:, None] * norms
        co += np.nan_to_num(movement)
    
    return co
    

# ===== GEOMETRY ===== #
def match_pairs(trico1, trico2, tco1, tco2):
    """Takes two sets of triangle Coords
    (left and right pairs of triangles)
    and rotates them so they are planar
    mirrors of each other aligned on
    the first edge."""
    
    M1 = matrix_from_verts(trico1)
    M2 = matrix_from_verts(trico2)
    
    NR1a = trico1 @ np.transpose(M1, axes=(0, 2, 1))
    NR1 = tco1 @ np.transpose(M1, axes=(0, 2, 1))
    NR2a = trico2 @ np.transpose(M2, axes=(0, 2, 1))
    NR2 = tco2 @ np.transpose(M2, axes=(0, 2, 1))
    
    eye = np.eye(3)
    eye[1] *= -1
    eyes = np.zeros((tco1.shape[0], 3, 3), dtype=np.float32)
    eyes[:] = eye
    NR3 = NR2 @ eyes
    NR3a = NR2a @ eyes    
    
    mids1 = NR1a[:, :2]
    vec = mids1[:, 1] - mids1[:, 0]
    new1 = mids1[:, 0] + vec * 0.5
    
    mids2 = NR3a[:, :2]
    vec2 = mids2[:, 1] - mids2[:, 0]
    new2 = mids2[:, 0] + vec2 * 0.5
        
    mid_dif = ((new2 - new1) * 0.5)[:, None]
    
    NR1 += mid_dif
    NR3 -= mid_dif
    
    return NR1, NR3


def rescale_sphere_aperture(vertices, original_radius, new_radius):
    """
    Change the radius of the sphere while keeping the angular (aperture) spacing of the points the same.
    
    Parameters:
    vertices (np.ndarray): Nx3 array of vertex coordinates in Cartesian (x, y, z).
    original_radius (float): The original radius of the sphere.
    new_radius (float): The desired new radius of the sphere.
    
    Returns:
    np.ndarray: Nx3 array of new vertex coordinates on the sphere with the new radius.
    """
    # Convert Cartesian to spherical coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Calculate the current radius for each point (distance from origin)
    current_radius = np.linalg.norm(vertices, axis=1)

    # Calculate the azimuth (theta) and elevation (phi) angles
    theta = np.arctan2(y, x)  # Azimuth angle (xy-plane)
    phi = np.arccos(z / current_radius)  # Elevation angle (z direction)

    # Calculate the ratio between the new and original radius
    scaling_factor = new_radius / original_radius
    
    # Adjust the angles (phi) based on the scaling factor (keeping theta constant)
    # New angular distance on the sphere is proportional to scaling_factor
    new_phi = np.arcsin(np.sin(phi) / scaling_factor)
    
    # Apply the new radius, keeping the same azimuth (theta) and adjusted elevation (phi)
    new_x = new_radius * np.sin(new_phi) * np.cos(theta)
    new_y = new_radius * np.sin(new_phi) * np.sin(theta)
    new_z = new_radius * np.cos(new_phi)
    
    # Return the new vertices in Cartesian coordinates
    return np.column_stack((new_x, new_y, new_z))


# get/set
def get_color_attributes(ob, color="Color"):
    col = np.empty((len(ob.data.vertices), 4), dtype=np.float32)
    ob.data.color_attributes[color].data.foreach_get("color", col.ravel())
    return col


def get_named_attribute(ob, name="Kitten Power"):
    ac = len(ob.data.attributes[name].data)
    arr = np.empty(ac, dtype=np.float32)
    ob.data.attributes[name].data.foreach_get('value', arr)
    return arr


def get_named_vector_attribute(ob, name="Kitten Locations"):
    ac = len(ob.data.attributes[name].data)
    arr = np.empty((ac, 3), dtype=np.float32)
    ob.data.attributes[name].data.foreach_get('vector', arr.ravel())
    return arr


def get_named_int_attribute(ob, name="Kitten Mana"):
    arr = np.empty(len(ob.data.vertices), dtype=np.int32)
    ob.data.attributes[name].data.foreach_get('value', arr)
    return arr


def get_named_int_attribute_mesh(mesh, name="Kitten Mana"):
    arr = np.empty(len(mesh.vertices), dtype=np.int32)
    mesh.attributes[name].data.foreach_get('value', arr)
    return arr


def set_named_vector_attribute(ob, arr, name="Kitten Normality", type="FLOAT_VECTOR", domain="POINT"):
    if name not in ob.data.attributes:
        ob.data.attributes.new(name, type, domain)
    ob.data.attributes[name].data.foreach_set('vector', arr.ravel())
            
    
def set_named_attribute(ob, arr, name="Kitten Agility", type="FLOAT", domain="POINT"):
    if name not in ob.data.attributes:
        ob.data.attributes.new(name, type, domain)
    ob.data.attributes[name].data.foreach_set('value', arr)
    

def set_named_int_attribute__(ob, arr, name="Kitten Mana", type="INT", domain="POINT"):
    if name not in ob.data.attributes:
        ob.data.attributes.new(name, type, domain)
    ob.data.attributes[name].data.foreach_set('value', arr)


def set_named_int_attribute(ob, arr, name="Kitten Mana", type="INT", domain="POINT"):
    # Check if we're in edit mode
    if ob.mode == 'EDIT':
        # Get the BMesh from edit mode
        bm = bmesh.from_edit_mesh(ob.data)
        
        # Get or create the custom data layer
        if name not in bm.verts.layers.int.keys():
            int_layer = bm.verts.layers.int.new(name)
        else:
            int_layer = bm.verts.layers.int[name]
        
        # Set values for each vertex
        for i, vert in enumerate(bm.verts):
            if i < len(arr):
                vert[int_layer] = arr[i]
        
        # Update the mesh
        bmesh.update_edit_mesh(ob.data)
    else:
        # Object mode - use your original method
        if name not in ob.data.attributes:
            ob.data.attributes.new(name, type, domain)
        ob.data.attributes[name].data.foreach_set('value', arr)


#x = get_named_attribute(bpy.context.object, name="Intensity")
#print(x.shape, "x_shape")


def set_color_attributes(ob, col, color="Color"):
    if not ob.data.color_attributes:
        ob.data.color_attributes.new(name="Kitten Stamina", type='BYTE_COLOR', domain='POINT')    
        color = 0
    ob.data.color_attributes[color].data.foreach_set("color", col.ravel())
    

# get/set
def hide_all(ob, hide=False):
    """Fast hide/unhide in object mode"""
    atts = [ob.data.vertices, ob.data.edges, ob.data.polygons]
    fun = np.zeros
    if hide:
        fun = np.ones
    for att in atts:
        c = len(att)
        arr = fun(c, dtype=np.bool)
        att.foreach_set('hide', arr)


# get/set
def get_eidx(ob):
    eidx = np.zeros((len(ob.data.edges), 2), dtype=np.int32)
    ob.data.edges.foreach_get("vertices", eidx.ravel())
    return eidx


# get/set
def get_selected_poly_verts(ob):
    """returns a list of lists of verts in each selected polygon.
    Works in any mode."""
    if ob.type != "MESH":
        return []

    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(ob.data)
        return [[v.index for v in f.verts] for f in bm.faces if f.select]

    return [[i for i in p.vertices] for p in ob.data.polygons if p.select]


# get/set
def get_poly_verts(ob):
    """returns a list of lists of verts in each polygon.
    Works in any mode."""
    if ob.type != "MESH":
        return []

    if ob.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(ob.data)
        return [[v.index for v in f.verts] for f in bm.faces]

    return [[i for i in p.vertices] for p in ob.data.polygons]


# get/set
def local_co(ob, obs):
    """Currently using deformed co to return coords
    of objects in obs in the local space of ob."""
    OBCOS = []
    WMDS = []
    for sob in obs:
        OBCOS += [deform_co(sob)]
        WMDS += [sob.matrix_world] # why monsters don't sweat
    
    mesh_matrix = ob.matrix_world
    local_matrix = np.linalg.inv(mesh_matrix) @ np.array(WMDS, dtype=np.float32)
    
    local_co = np.empty((0, 3), dtype=np.float32)    
    for i in range(len(OBCOS)):
        ob_space = OBCOS[i] @ local_matrix[i][:3, :3].T
        ob_space += local_matrix[i][:3, 3]
        local_co = np.concatenate((local_co, ob_space), axis=0)
    
    return local_co


# get/set
def deform_co(ob, return_prox=False, triangulate=False, return_tobm=False):
    """Uses a list of mods that only deform,
    creates a proxy turning of every other
    mod then returns the proxy coords."""
    deform_mods = ['ARMATURE',
                   'CAST',
                   'CURVE',
                   'DISPLACE',
                   'HOOK',
                   'LAPLACIANDEFORM',
                   'LATTICE',
                   'MESH_DEFORM',
                   'SHRINKWRAP',
                   'SIMPLE_DEFORM',
                   'SMOOTH',
                   'CORRECTIVE_SMOOTH',
                   'LAPLACIANSMOOTH',
                   'SURFACE_DEFORM',
                   'WARP',
                   'WAVE',
                   'VOLUME_DISPLACE',
                   'CLOTH',
                   'SOFT_BODY',
                   'VERTEX_WEIGHT_EDIT',
                   'VERTEX_WEIGHT_MIX',
                   'VERTEX_WEIGHT_PROXIMITY',
                   ]        
    
    display = [m.show_viewport for m in ob.modifiers]
    for m in ob.modifiers:
        if m.type not in deform_mods:
            m.show_viewport = False
    
    prox = prox_object(ob)#, triangulate=True)
    
    if triangulate:
        tobm = get_bmesh(prox, refresh=True)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
    
    co = get_co(prox)
    
    for e, m in enumerate(ob.modifiers):
        m.show_viewport = display[e]
    
    if return_prox:
        return co, prox
    if return_tobm:
        return co, tobm
    
    return co


# geometry
def closest_points_edge(vec, origin, p):
    '''Returns the location of the points on the edge'''
    vec2 = p - origin
    d = np.einsum('j,ij->i', vec, vec2) / (vec @ vec)
    cp = origin + vec * d[:, None]
    return cp, d


# =============== #
# GEOMETRY ====== #
# =============== #

# geometry
def circular_order(co, v1, v2, center=None, edges=False, convex=False, normal=None):
    """Return an array that indexes the points in circular order.
    v1 and v2 must be perpindicular and their normal defines the axis.
    if edges is True, return the edges to connect the points"""
    if co.shape[0] == 0:
        return
    #if center is None:
    center = np.mean(co, axis=0)
    if convex:
        center_vecs = co - center
        center_dots = np.einsum('ij,ij->i', center_vecs, center_vecs)
        max = np.argmax(center_dots)
        out_vec = center_vecs[max]
        cross = np.cross(out_vec, normal)
        con_set = [max]
        point = max 
        
        for i in range(co.shape[0]):
            spread = co - co[point]
            h = np.einsum('ij,ij->i', spread, spread)    
            Uspread = np.nan_to_num(spread / np.sqrt(h)[:, nax])
            dots = np.einsum('j,ij->i', cross, Uspread)
            new = np.argmax(dots)
            if new == point:
                new = np.argsort(dots)[-2] # for when the point gets narcissistic and finds itself
            if new == max:
                break

            con_set.append(new)
            cross = co[new] - co[point]
            point = new
 
        idxer = np.arange(len(con_set))
        eidx = np.append([idxer],[np.roll(idxer, -1)], 0).T       

        return con_set, eidx, center
    
    count = co.shape[0]
    idxer = np.arange(count)
    on_p1, center_vecs = cp_scalar(v1, center, co, False, True) # x_vec off center
    pos_x = on_p1 > 0
    co_pos = co[pos_x]
    co_neg = co[-pos_x]
    p_on_p2 = cp_scalar(v2, center, co_pos, True)
    n_on_p2 = cp_scalar(v2, center, co_neg, True)
    p_y_sort = np.argsort(p_on_p2)
    n_y_sort = np.argsort(n_on_p2)
    order = np.append(idxer[pos_x][p_y_sort], idxer[-pos_x][n_y_sort][::-1])
            
    idxer = np.arange(len(order))
    eidx = np.append([idxer],[np.roll(idxer, -1)], 0).T

    return order, eidx


# ===== GEOMETRY ===== #
def smooth_shape_blend(ob, target_shape, blend_value, g_steps, axis=[0, 1, 2]):

    mode = ob.mode
    edit = False
    if mode == "EDIT":
        edit = True
        bpy.ops.object.mode_set()

    keys = ob.data.shape_keys
    obm = get_bmesh(ob)
    sel, steps = grow(obm, iters=g_steps, compute_steps=True)
    vc = len(ob.data.vertices)
    active = ob.active_shape_key
    co = np.empty((vc, 3), dtype=np.float32)
    keys.key_blocks['PC Smooth Base'].data.foreach_get('co', co.ravel())
    tco = np.empty((vc, 3), dtype=np.float32)
    keys.key_blocks[target_shape].data.foreach_get('co', tco.ravel())
    
    mult = np.linspace(0, 1, g_steps + 1)[::-1][:, None] * blend_value    
    move = (tco - co) * mult[steps]
    mask = np.ones(3, dtype=bool)
    
    mask[axis] = False
    move[:, mask] = 0.0
    co[sel] += move[sel]
    
    active.data.foreach_set("co", co.ravel())
    ob.data.update()
    
    if edit:
        bpy.ops.object.mode_set(mode="EDIT")


# ===== GEOMETRY ===== #
def loop_order(ob):
    """takes an object consisting of a single loop of edges and gives the order"""
    obm = get_bmesh(ob)
    obm.edges.ensure_lookup_table()
    e = obm.edges[0]
    v = e.verts[0]
    order = []
    for i in range(len(obm.edges)):
        other = e.other_vert(v)
        order.append(other.index)
        e = [ed for ed in v.link_edges if ed != e][0]
        v = [ve for ve in e.verts if ve != other][0]
    return order


# ===== GEOMETRY ===== #
def mag_set(mag, v2):    
    '''Applys the magnitude of v1 to v2'''
    d1 = mag ** 2
    d2 = v2 @ v2
    div = d1/d2
    return v2 * np.sqrt(div)


# ===== GEOMETRY ===== #
def in_line_bounds(vec, origin, p):
    '''Returns a bool array indicating if points
    are in the range of the start and end of a vector'''
    vec2 = p - origin
    d = np.einsum('j,ij->i', vec, vec2)
    vd = vec @ vec 
    bool = (d >= 0) & (d <= vd)    
    return bool


# ===== GEOMETRY ===== #
def analyze_mesh_quality(vertices, faces, max_edge_length):
    """
    Analyze the quality of the generated mesh.
    
    Parameters:
    vertices: np.array of shape (N, 3) containing vertex coordinates
    faces: np.array of shape (M, 3) containing vertex indices
    max_edge_length: float, the maximum edge length used to generate the mesh
    
    Returns:
    dict containing mesh quality metrics
    """
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Calculate edge lengths
    edges = np.array([
        np.sqrt(np.sum((v1 - v0) ** 2, axis=1)),
        np.sqrt(np.sum((v2 - v1) ** 2, axis=1)),
        np.sqrt(np.sum((v0 - v2) ** 2, axis=1))
    ])
    
    # Calculate face normals
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]
    
    # Calculate face areas
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    return {
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'min_edge_length': np.min(edges),
        'max_edge_length': np.max(edges),
        'mean_edge_length': np.mean(edges),
        'std_edge_length': np.std(edges),
        'total_area': np.sum(face_areas),
        'face_normals': face_normals
    }


# ===== GEOMETRY ===== #
def find_nearest_neighbors(query_point, points, k=5, radius=None, tree=None, remove_qp=True):
    """
    Find nearest neighbors for a query point within a set of points in 3D space.
    
    Parameters:
    -----------
    query_point : array-like, shape (3,)
        The point for which to find neighbors
    points : array-like, shape (N, 3)
        The set of N points in 3D space
    k : int, optional (default=5)
        Number of nearest neighbors to return
    radius : float, optional (default=None)
        If specified, return only points within this radius.
        If both k and radius are specified, returns up to k points within the radius.
    
    Returns:
    --------
    indices : array of indices of the nearest points
    distances : array of distances to the nearest points
    """
    
    # Build KD Tree (this is the expensive operation, but only needs to be done once)
    if tree is None:
        tree = KDTree(points)
    
    # Query the KD Tree
    if radius is not None:
        # For fixed-radius search
        indices = tree.query_ball_point(query_point, radius)
        if k is not None and len(indices) > k:
            # If we have more points than k, find the k closest ones
            distances = np.linalg.norm(points[indices] - query_point, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            return np.array(indices)[nearest_indices], distances[nearest_indices]
        else:
            distances = np.linalg.norm(points[indices] - query_point, axis=1)
            return indices, distances
    else:
        # For k-nearest neighbors
        distances, indices = tree.query(query_point, k=k)
        if remove_qp: # assumes the query_point is in the neighbor search
            clean = distances > 0.0000001
            return indices[clean], distances[clean]    
        return indices, distances


# ===== GEOMETRY ===== #
def cp_scalar(vec, origin, p, unitize=False):
    '''Returns the dot that would put the point on the edge.
    Useful for sorting the order of verts if they were
    projected to the closest point on the edge'''
    vec2 = p - origin
    if unitize:
        vec2 = vec2 / np.sqrt(np.einsum('ij,ij->i', vec2, vec2))[:, nax]
    d = np.einsum('j,ij->i', vec, vec2)
    return d
    

# ===== GEOMETRY ===== # 2d
def edge_edge_intersect_2d(a1,a2, b1,b2, intersect=False):
    """2d line intersect for two edges"""    
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = da[::-1] * np.array([1,-1])
    denom = dap @ db
    num = dap @ dp
    scale = (num / denom)
    loc = b1 + db * scale
    
    if intersect:
        check_1 = (scale >= 0.0) & (scale <= 1.0)
        vec = loc - a1
        dot = vec @ vec
        ed = da @ da
        edv = vec @ da
        check_2 = (edv >= 0.0) & (dot <= ed)
        return loc, check_1 & check_2
    else:
        return loc


# ===== GEOMETRY ===== # 2d
def edge_edges_intersect_2d(a1,a2, b1,b2, intersect=False):
    """2d line intersect for one edge against multiple edges"""
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = da[::-1] * np.array([1,-1])
    denom = db @ dap
    num =  dp @ dap
    scale = (num / denom)
    loc = b1 + db * scale[:, None]
    
    if intersect:
        check_1 = (scale >= 0.0) & (scale <= 1.0)
        vec = loc - a1
        dot = np.einsum('ij,ij->i', vec, vec)
        ed = da @ da
        edv = vec @ da
        check_2 = (edv >= 0.0) & (dot <= ed)
        return loc, check_1 & check_2
    else:
        return loc


def edges_edges_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''2d line intersect for two groups of edges.''' 
    # this fails in certain cases.
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = da[:, ::-1] * np.array([1, -1])
    denom = np.einsum('ij,ij->i', dap, db)    
    num = np.einsum('ij,ij->i', dap, dp)
    scale = (num / denom)

    if intersect:
        return b1 + db * scale[:, None], (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale[:, None]


def visualize_points(co, name="VisEmpties_", scale=0.1):
    
    i = -1
    for i, c in enumerate(co):
        if name + str(i) in bpy.data.objects:
            empty = bpy.data.objects[name + str(i)]
        else:
            empty = bpy.data.objects.new(name + str(i), None)
            bpy.context.collection.objects.link(empty)
            empty.empty_display_type = "SPHERE"
        
        empty.location = co[i]
        empty.scale = np.array([scale] * 3)
            
    vis_empties = [ob.name for ob in bpy.data.objects if ob.name.startswith(name)]        
    for ob_name in vis_empties:
        number = int(ob_name.split("_")[1])
        if number > i:
            bpy.data.objects.remove(bpy.data.objects[ob_name])


def visualize_forces(start, end=None, name="viz", offset=0, offset_axis=0):
    new = False
    if name in bpy.data.objects:
        ob = bpy.data.objects[name]    
        vc = len(ob.data.vertices)
        if vc != start.shape[0] * 2:
            new = True
    else:    
        new = True
    
    if end is None:
        end = start + np.array([0.0,0.0,0.1])
    
    verts = np.append(start, end, axis=0)
    if new:
        if name in bpy.data.objects:
            mesh = ob.data
            bpy.data.objects.remove(ob)
            bpy.data.meshes.remove(mesh)
        
        eidx = np.empty((start.shape[0], 2), dtype=np.int32)
        idxr = np.arange(start.shape[0])
        eidx[:, 0] = idxr
        eidx[:, 1] = idxr + (start.shape[0])
        ob = ob_from_py_data(verts, faces=[], edges=eidx, name=name)
        return ob
    verts[:, offset_axis] += offset
    ob.data.vertices.foreach_set('co', verts.ravel())
    ob.data.update()
    return ob


def test_edge():
    ob = bpy.context.object
    co = get_co(ob)[:, :2]
    eidx = get_eidx(ob)
    a1, a2 = co[eidx[0]][0], co[eidx[0]][1]
    b1, b2 = co[eidx[1:][:, 0]], co[eidx[1:][:, 1]]
    inter, in_range = edge_edges_intersect_2d(a1,a2, b1,b2, intersect=True)
    bpy.data.objects['eee'].location.xy = inter[0]
    bpy.data.objects['eee1'].location.xy = inter[1]
    bpy.data.objects['eee2'].location.xy = inter[2]
    print(in_range, "in range??")
    

def replot_orthagonal_mix(ph, use_cp=False):

    ### ===== MAKE X ORTHAGONAL TO Y ===== ###
    x_co = ph.current_co[ph.x_vidx]
    
    if use_cp:
        yeco = ph.current_co[ph.eidx]
        yvecs = yeco[:, 1] - yeco[:, 0]
        cp, d = closest_points_edges(yvecs, yeco[:, 0], x_co[1::2])
        
        vecs = x_co[1::2] - cp
        u_v = u_vecs(vecs)
        return u_v

    vecs = x_co[1::2] - x_co[0::2]
    u_v = u_vecs(vecs)
    return u_v
    

def ___vecs_to_matrix(ph):
    mesh_edge_co = ph.current_co[ph.eidx]
    mesh_vecs = mesh_edge_co[:, 1] - mesh_edge_co[:, 0]
    mesh_u_vecs = u_vecs(mesh_vecs)
    u_x_vecs = replot_orthagonal_mix(ph)
    z_vecs = np.cross(u_x_vecs, mesh_u_vecs)    
    ph.physics_m3[:, :, 0] = u_x_vecs
    ph.physics_m3[:, :, 1] = mesh_u_vecs
    ph.physics_m3[:, :, 2] = z_vecs
    return mesh_edge_co, u_x_vecs, mesh_u_vecs, z_vecs


def make_x_orthagonal(ph):
    
    y_eidx = ph.current_co[ph.eidx]
    pivots = np.mean(y_eidx, axis=1, out=ph.pivots)
    x_co = ph.current_co[ph.x_vidx]
    
    yeco = ph.current_co[ph.eidx]
    yvecs = yeco[:, 1] - yeco[:, 0]
    cp_0, d = closest_points_edges(yvecs, yeco[:, 0], x_co[0::2])
    cp, d = closest_points_edges(yvecs, yeco[:, 0], x_co[1::2])
    
    vecs1 = x_co[1::2] - cp
    vecs2 = x_co[0::2] - cp_0
    vecs = (vecs1 + -vecs2) * 0.5 # to prevent distorting the angle
    
    u_x_vecs = u_vecs(vecs)
    half = u_x_vecs * 0.5
    ph.current_co[ph.yvc::2] = pivots - half
    ph.current_co[ph.yvc + 1::2] = pivots + half
    return u_x_vecs, pivots


def vecs_to_matrix(ph, cp=False, skip=False):

    if cp:
        u_x_vecs, pivots = make_x_orthagonal(ph)
    else:    
        x_co = ph.current_co[ph.x_vidx]
        vecs = x_co[1::2] - x_co[0::2]
        u_x_vecs = u_vecs(vecs)

    mesh_edge_co = ph.current_co[ph.eidx]
    mesh_vecs = mesh_edge_co[:, 1] - mesh_edge_co[:, 0]
    mesh_u_vecs = u_vecs(mesh_vecs)    
    z_vecs = np.cross(u_x_vecs, mesh_u_vecs)

    if skip:
        ph.physics_m3[:, 0] = u_x_vecs
        ph.physics_m3[:, 1] = mesh_u_vecs
        ph.physics_m3[:, 2] = z_vecs
        return
    
    ph.physics_m3[:, :, 0] = u_x_vecs
    ph.physics_m3[:, :, 1] = mesh_u_vecs
    ph.physics_m3[:, :, 2] = z_vecs
    return mesh_edge_co, u_x_vecs, mesh_u_vecs, z_vecs 


def replot_orthagonal(ph, skip_mean=False, new_vecs=None):
    """Find the mean of x edge middle
    and corresponding physics mesh middle.
    After that make x_edges orthagonal."""

    ### ===== X THINGY MID MEAN ===== ###
#    if not skip_mean:
#    e_vecs = ph.x_co[1::2] - ph.x_co[::2]
#    x_mid = ph.x_co[::2] + (e_vecs * ph.x_factor)
#    mix_mid = (ph.current_co[ph.mesh_bone_idx] + x_mid) * 0.5
#    ph.x_mid = mix_mid
#    ph.current_co[ph.mesh_bone_idx] = mix_mid
    #else:
        #ph.x_mid = ph.current_co[ph.mesh_bone_idx]

    ph.x_mid = ph.current_co[ph.mesh_bone_idx]
    
    ### ===== MAKE X ORTHAGONAL TO Y ===== ###
    eco = ph.current_co[ph.eidx]
    vecs = eco[:, 1] - eco[:, 0]
    
    if new_vecs is None:
        cp, d = closest_points_edges(vecs=vecs, origins=ph.x_mid, p=ph.x_co[::2])
        cp_vec = cp - ph.x_co[::2]
        ucp = u_vecs(cp_vec)

        ph.x_co[::2] = ph.x_mid - (ucp * ph.x_factor)
        ph.x_co[1::2] = ph.x_mid + (ucp * (1 - ph.x_factor))
        return ucp
    
    ph.x_co[::2] = ph.x_mid - (new_vecs * ph.x_factor)
    ph.x_co[1::2] = ph.x_mid + (new_vecs * (1 - ph.x_factor))
    



#=======================#
# MATRIX ---------------#
#=======================#
def m3_to_axis_angle(m3):

    angle = np.arccos((np.trace(m3) - 1) / 2.0)
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    dif = m3[[2, 0, 1], [1, 2, 0]] - m3[[1, 2, 0], [2, 0, 1]]

    axis = 1 / (2 * np.sin(angle)) * dif
    return axis * angle


def axis_angle_to_m3(axis_angle):

    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle if angle != 0 else np.array([0.0, 0.0, 0.0])

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    m3 = np.array([
        [t * axis[0]**2 + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1]**2 + c, t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2]**2 + c]
    ], dtype=np.float32)

    return m3


def interpolate_m3(mat1, mat2, factor):
    axis_angle1 = m3_to_axis_angle(mat1)
    axis_angle2 = m3_to_axis_angle(mat2)
    interpolated_axis_angle = (1 - factor) * axis_angle1 + factor * axis_angle2
    interpolated_matrix = axis_angle_to_m3(interpolated_axis_angle)
    return interpolated_matrix

# Example usage:
# r2 = interpolate_m3(m1, m2, 0.5)
# M3[:3, :3] = r2
# e2.matrix_world = M3.T
# chat gippity came up with this code and it does not
#   rotate smoothly (probably when the normal of the cross
#   flips.) It's jerky. Two step quat approach will probably
#   be better.


#=======================#
# CP SPECIFIC ----------#
#=======================#
def reset_pose(ar, ph, use_shape=False):

    mix_mesh = ar.CP_props.mix_mesh

    if ph:
        if mix_mesh:
            active_shape = "Current"
            if use_shape:
                active_shape = mix_mesh.active_shape_key.name

            active_co = get_shape_co_mode(ob=mix_mesh, co=None, key=active_shape)
            
            ph.current_co[:] = active_co
            
            mesh_bone_co = active_co[ph.mesh_bone_idx]
            vecs_to_matrix(ph, cp=True)
            
            set_ar_m3_world(ar, ph.physics_m3, locations=mesh_bone_co, return_quats=False)   
            
    else:
        loc = (0.0, 0.0, 0.0)
        rot = (1.0, 0.0, 0.0, 0.0)
        scale = (1.0, 1.0, 1.0)
        for bo in ar.pose.bones:
            bo.location = loc
            bo.rotation_quaternion = rot
            bo.scale = scale


def get_ar_matrix_world(ar, m3=False):
    """Returns whirled matrix."""
    world_matrix = np.array([bo.matrix for bo in ar.pose.bones], dtype=np.float32)
    if m3:
        return world_matrix[:, :3, :3]
    return world_matrix


#### quaternions ####
def q_rotate(co, w, axis):
    """Takes an N x 3 numpy array and returns that array rotated around
    the axis by the angle in radians w. (standard quaternion)"""    
    move1 = np.cross(axis, co)
    move2 = np.cross(axis, move1)
    move1 *= w
    return co + (move1 + move2) * 2


def get_quat(rad, axis, normalize=False):
    if normalize:
        axis = axis / np.sqrt(axis @ axis)
    theta = (rad * 0.5)
    w = np.cos(theta)
    q_axis = axis * np.sin(theta)
    return w, np.nan_to_num(q_axis)


def rotate_Q_matrix(verts, axis, angle, origin=None):
    if origin:
        verts -= origin
    w, axis = get_quat(axis, angle)
    M = np.eye(3, dtype=np.float32)
    rotm = q_rotate(M, w, axis)
    rotv = verts @ rotm
    if origin:
        rotv += origin
    return rotv


#### quaternions ####
def set_ar_m3_world(ar, m3, locations=None, return_quats=False):
    """Sets the world rotation correctly
    in spite of parent bones. (and constraints??)"""
    if return_quats:
        quats = np.empty((m3.shape[0], 4), dtype=np.float32)
    for i in range(len(ar.pose.bones)):
        bpy.context.view_layer.update()
        arm = ar.pose.bones[i].matrix
        nparm = np.array(arm)
        nparm[:3, :3] = m3[i]
        if locations is not None:
            nparm[:3, 3] = locations[i]
        #mat = MAT(nparm)
        mat = nparm.T
        if return_quats:
            quats[i] = mat.to_quaternion()
        ar.pose.bones[i].matrix = mat
    
    bpy.context.view_layer.update()
    if return_quats:    
        return quats


class Nc():
    pass


def update_node_sizes(ph):
    rig = ph.physics_rig

    head_node_sizes = np.array([b.CP_props.head_node_size for b in rig.pose.bones], dtype=np.float32)
    tail_node_sizes = np.array([b.CP_props.tail_node_size for b in rig.pose.bones], dtype=np.float32)
    
    ph.node_sizes = np.zeros(ph.yvc, dtype=np.float32)
    np.add.at(ph.node_sizes, ph.mesh_bone_idx, head_node_sizes)
    np.add.at(ph.node_sizes, ph.mesh_bone_tail_idx, tail_node_sizes)    

    idxc = ph.mesh_bone_idx.shape[0]
    counts = np.zeros(ph.yvc, dtype=np.float32)
    np.add.at(counts, ph.mesh_bone_idx, np.ones(idxc, dtype=np.float32))
    np.add.at(counts, ph.mesh_bone_tail_idx, np.ones(idxc, dtype=np.float32))

    ph.node_sizes = (ph.node_sizes / counts)[:, None]


def update_node_friction(ph):
    rig = ph.physics_rig
    head_friction = np.array([b.CP_props.head_friction for b in rig.pose.bones], dtype=np.float32)
    tail_friction = np.array([b.CP_props.tail_friction for b in rig.pose.bones], dtype=np.float32)

    ph.node_friction = np.zeros(ph.yvc, dtype=np.float32)
    np.add.at(ph.node_friction, ph.mesh_bone_idx, head_friction)
    np.add.at(ph.node_friction, ph.mesh_bone_tail_idx, tail_friction)    

    idxc = ph.mesh_bone_idx.shape[0]
    counts = np.zeros(ph.yvc, dtype=np.float32)
    np.add.at(counts, ph.mesh_bone_idx, np.ones(idxc, dtype=np.float32))
    np.add.at(counts, ph.mesh_bone_tail_idx, np.ones(idxc, dtype=np.float32))

    ph.node_friction = (ph.node_friction / counts)[:, None]


def refresh_collision_objects(ph):
  
    validate_references(ph)

    if len(bpy.context.scene['Ce_collider_objects']) == 0:
        return

    nc = Nc()

    nc.yco = ph.current_co[:ph.yvc] # this is a view that will overwrite ph.current_co
    nc.start_yco = np.copy(nc.yco)
    nc.joined_yco = np.empty((ph.yvc, 2, 3), dtype=np.float32)
    nc.joined_yco[:, 0] = nc.yco
    nc.joined_yco[:, 1] = nc.start_yco
    
    update_node_sizes(ph)
    update_node_friction(ph)
    # get all collider coords in rig space
    WMDS = [] # worlds of matrix destruction
    OBCOS = [] # objective concerns about the letter "S"
    for ob in bpy.context.scene['Ce_collider_objects']:
        OBCOS += [absolute_co(ob, world=False)]
        WMDS += [ob.matrix_world] # why monsters don't sweat
    
    mesh_matrix = ph.mix_mesh.matrix_world
    local_matrix = np.linalg.inv(mesh_matrix) @ np.array(WMDS, dtype=np.float32)
    
    nc.rig_space_co = np.empty((0, 3), dtype=np.float32)    
    for i in range(len(OBCOS)):
        rig_space_co = OBCOS[i] @ local_matrix[i][:3, :3].T
        rig_space_co += local_matrix[i][:3, 3]
        nc.rig_space_co = np.concatenate((nc.rig_space_co, rig_space_co), axis=0)
                
    nc.start_rig_space_co = np.copy(nc.rig_space_co)
    # ------------------------------------

    tridex_offset = [0] + [len(ob.data.vertices) for ob in bpy.context.scene['Ce_collider_objects']]
    
    nc.collider_v_count = np.sum(tridex_offset)
    nc.tridex_offset = np.cumsum(tridex_offset)
    nc.collider_f_count = np.sum([len(ob.data.polygons) for ob in bpy.context.scene['Ce_collider_objects']])
    
    nc.tridex = np.empty((0, 3), dtype=np.int32)
    nc.t_eidx = np.empty((0, 2), dtype=np.int32)
    nc.triangle_friction = np.empty((0, 1), dtype=np.float32)
    nc.edge_friction = np.empty((0, 1), dtype=np.float32)
    nc.vertex_friction = np.empty((0, 1), dtype=np.float32)
    
    # need the edge idx for the three edges for each tri
    nc.three_edges = np.empty((0, 3), dtype=np.int32)
    nc.edge_normal_keys = np.empty((0, 2), dtype=np.int32)
    
    edge_offset = 0
    triangle_offset = 0
    for e, ob in enumerate(bpy.context.scene['Ce_collider_objects']):
        prox = prox_object(ob)
        tridex, t_eidx, three_edges, edge_normal_keys = get_tridex(prox, free=True)
        nc.tridex = np.concatenate((nc.tridex, tridex + nc.tridex_offset[e]), axis=0)
        nc.t_eidx = np.concatenate((nc.t_eidx, t_eidx + nc.tridex_offset[e]), axis=0)
        nc.three_edges = np.concatenate((nc.three_edges, three_edges + edge_offset), axis=0)
        edge_offset += t_eidx.shape[0]
        nc.edge_normal_keys = np.concatenate((nc.edge_normal_keys, edge_normal_keys + triangle_offset), axis=0)
        triangle_offset += tridex.shape[0]
        tw = get_vertex_weights(prox, "CP_friction", default=0.0)[tridex] * ob.CP_props.object_friction
        ew = get_vertex_weights(prox, "CP_friction", default=0.0)[t_eidx] * ob.CP_props.object_friction
        vw = get_vertex_weights(prox, "CP_friction", default=0.0)[:, None] * ob.CP_props.object_friction
        mix_tw = (np.sum(tw, axis=1) / 3)[:, None]
        nc.triangle_friction = np.concatenate((nc.triangle_friction, mix_tw), axis=0)
        mix_ew = (np.sum(ew, axis=1) / 2)[:, None]
        nc.edge_friction = np.concatenate((nc.edge_friction, mix_ew), axis=0)
        nc.vertex_friction = np.concatenate((nc.vertex_friction, vw), axis=0)    
    
    nc.tri_edge_bool = np.zeros(ph.current_co.shape[0], dtype=bool)
    
    nc.tri_co = nc.rig_space_co[nc.tridex]
    ph.triangle_friction = nc.triangle_friction
        
    nc.tri_normals = get_tri_normals(nc.tri_co, normalize=True)
    
    nc.joined_tri_co = np.empty((nc.tridex.shape[0], 6, 3), dtype=np.float32)
    
    nc.joined_tri_co[:] = -5
    
    nc.joined_tri_co[:, :3] = nc.tri_co
    nc.joined_tri_co[:, 3:] = nc.tri_co
    ph.joined_tri_co = nc.joined_tri_co
    
    nc.tid = np.arange(nc.tridex.shape[0])
    nc.eid = np.arange(ph.yvc)
    nc.box_max = 375000
    ph.nc = nc
    ph.hit_eidx = np.zeros(0, dtype=bool)

    return nc


def rig_reset(data=None, ph=None):
    # stuff
    # just a place to put this while testing
    bpy.context.scene.CP_props.skip_handler = False
    # --------------------------------------

    if ph is None:
        
        ob = bpy.context.object
        rig = ob # or not
        if not ob:
            print("tried to reset with no active object")
            return
        
        if ob.type == "MESH":
            if not ob.CP_props.physics_rig:
                print("tried to reset from mesh with no physics_rig")
                return
            mix_mesh = ob
            rig = ob.CP_props.physics_rig
        
        if ob.type == "ARMATURE":
            if ob.CP_props.physics_rig:
                rig = ob.CP_props.physics_rig
            if ob.CP_props.pose_target:
                rig = ob        
        
    else:
        validate_references(ph)
        rig = ph.physics_rig
    
    CP = rig.CP_props
            
    use_shape = CP.reset_to_active_shape
                    
    alt_target = CP.alternate_target

    mix_mesh = CP.mix_mesh
    if mix_mesh:
        
        if not ph:
            if data:
                if CP.data_key in data:
                    ph = data[CP.data_key]
                else:
                    print("ph not found in data in rig_reset")
        
        if ph:
            shape = 'Current'
            if use_shape:
                shape = mix_mesh.active_shape_key.name
                
            co_to_shape(mix_mesh, co=None, key="Current")
            co_to_shape(mix_mesh, co=None, key="Target")
            co_to_shape(mix_mesh, co=None, key="Basis")
            
            vc = len(mix_mesh.data.vertices)
            ph.current_co = np.empty((vc, 3), dtype=np.float32)
            get_shape_co_mode(ob=mix_mesh, co=ph.current_co, key=shape)
            ph.velocity = np.zeros((vc, 3), dtype=np.float32)
            ph.vel_start = np.empty((vc, 3), dtype=np.float32)
            ph.vel_start[:] = ph.current_co
            #ph.previous_co[:] = ph.current_co
            ph.target_m3 = get_ar_matrix_world(ph.pose_target, m3=True)
            ph.physics_m3 = get_ar_matrix_world(rig, m3=True)
            refresh_collision_objects(ph)
            ph.hit_idx = []
            ph.pre_tidx = np.empty(0, dtype=np.int32)
            ph.eco_idx = np.empty(0, dtype=np.int32)
            ph.tco_idx = np.empty(0, dtype=np.int32)
            ph.friction = np.empty((0, 1), dtype=np.float32)
            
            set_shape_co(mix_mesh, "Current", ph.current_co)
            mix_mesh.data.update()
                
            ### ----- UPDATE RIG ----- ###
            mesh_bone_co = ph.current_co[ph.mesh_bone_idx]
            vecs_to_matrix(ph, cp=True)
            
            alt_target = rig.CP_props.alternate_target
            if alt_target:
                set_ar_m3_world(alt_target, ph.physics_m3, locations=mesh_bone_co, return_quats=False)   
            else:    
                set_ar_m3_world(ph.physics_rig, ph.physics_m3, locations=mesh_bone_co, return_quats=False)   
            
            return

    ar = rig
    if alt_target:
        ar = alt_target

    loc = (0.0, 0.0, 0.0)    
    rot = (1.0, 0.0, 0.0, 0.0)
    scale = (1.0, 1.0, 1.0)
    for bo in ar.pose.bones:
        bo.location = loc
        bo.rotation_quaternion = rot
        bo.scale = scale


def get_circle_points(n_points, radius=1):
    """plot points around a circle"""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Calculate x and y coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    return x, y


def test_curves():
    ob = bpy.context.object
    co = get_co(ob)
    scalars = co[:, 0]
    scalars = np.linspace(0,1,co.shape[0])# ** np.pi
    co[:,0] = scalars
    print(scalars, "this")
    #x = 1 / np.max(scalars)

    x, y = get_circle_points(co.shape[0], radius=1)
    
    co[:,2] = x
    co[:,0] = y
    #co[:,0] /= map
    
    
    ob.data.vertices.foreach_set('co', co.ravel())
    ob.data.update()
    
    
#test_curves()
#=======================#
# UV MAPS --------------#
#=======================#
def get_tri_loop_indices(ob):
    """Retrieve loop indices for each triangle in the mesh, analogous to get_tridex_3."""
    triangles = np.empty((len(ob.data.loop_triangles), 3), dtype=np.int32)
    ob.data.loop_triangles.foreach_get('loops', triangles.ravel())
    return triangles


def get_uv(ob, uvm=0):
    uvc = len(ob.data.loops)
    uv = np.zeros((uvc, 2), dtype=D_TYPE_F)
    ob.data.uv_layers[uvm].data.foreach_get('uv', uv.ravel())
    return uv
    

def get_loops(ob):
    loops = np.empty(len(ob.data.loops), dtype=D_TYPE_I)
    ob.data.loops.foreach_get('vertex_index', loops)    
    return loops


def uv_to_co(ob, uvm=0):
    loops = get_loops(ob)
    if len(loops) == 0:
        check_faces(ob)
        unwrap_object(ob)
        loops = get_loops(ob)
    uv = get_uv(ob, uvm)
    uni, idx = np.unique(loops, return_index=True)
    co = np.zeros((len(ob.data.vertices), 3), dtype=D_TYPE_F)
    co[:, :2] = uv[idx]
    return co


def generate_small_random_array(size):
    min_val = np.finfo(np.float32).tiny
    base_val = min_val * 1000
    multipliers = np.random.uniform(1, 10, size)
    return (base_val * multipliers).astype(np.float32)


def uv_shape(ob, uvm=0, shape=["Basis", "uv_shape"], skip=False):
    
    if not skip:    
        manage_shapes(ob, shape)
    
    if ob.data.is_editmode:
        ob.update_from_editmode()

    sco = get_co(ob)    
    smean = np.mean(sco, axis=0)
    
    uvc = uv_to_co(ob, uvm)

    eidx = get_eidx(ob)
    svecs = sco[eidx[:, 1]] - sco[eidx[:, 0]]
    savd = np.mean(measure_vecs(svecs))
    
    uvc_vecs = uvc[eidx[:, 1]] - uvc[eidx[:, 0]]
    avd = np.mean(measure_vecs(uvc_vecs))

    scale = savd / avd
    
    uvc *= scale
    uvm = np.mean(uvc, axis=0)
    dif = smean - uvm
    uvc += dif
    
    if skip:
        return uvc
    
    key = ob.data.shape_keys.key_blocks[shape[-1]]
    key.value = 1.0
    key.data.foreach_set('co', uvc.ravel())
    ob.data.update()
    

def set_uv(ob, uvco, uvm=0):
    ob.data.uv_layers[uvm].data.foreach_set('uv', uvco.ravel())
    ob.data.update()


def copy_uvs():
    ob = bpy.context.object
    sob = [obs for obs in bpy.data.objects if obs.select_get() & (obs != ob)]
    uv = get_uv(ob)
    for obs in sob:
        set_uv(obs, uv)
    

def co_to_uv(ob, cam):
    loops = get_loops(ob)
    co = get_co(ob)
    co = co_to_matrix_space(co, ob, cam)

    x_res = bpy.context.scene.render.resolution_x
    y_res = bpy.context.scene.render.resolution_y
    
    if x_res > y_res:
        div = x_res / y_res
        axis = 1
    else:
        div = y_res / x_res
        axis = 0

    if cam.data.type == "PERSP":
        
        # Focal length and sensor width in meters
        f = cam.data.lens / 1000  # mm to meters
        w = cam.data.sensor_width / 1000  # mm to meters
        if axis == 1:    
            h = w / div
        else:
            h = w
            w = w / div    

        # Perspective projection
        x_proj = (f * co[:, 0]) / (-co[:, 2])
        y_proj = (f * co[:, 1]) / (-co[:, 2])
        
        # Map to UV space [0, 1]
        u = x_proj / w + 0.5
        v = y_proj / h + 0.5
        
        # Assign to coordinate array
        co[:, 0] = u
        co[:, 1] = v
        
        return co[:, :2][loops], co[:, 2], loops
        
    xy = co[:, :2]
    xy[:, axis] *= div    
    u_uv = xy + np.array([0.5, 0.5])
    uv = u_uv[loops]
    return uv, co[:, 2], loops
    

def exclude_xy(triangles, square_corner=(1.0, 1.0), square_size=1.0):

    x_min, y_min = square_corner
    x_max, y_max = x_min + square_size, y_min + square_size
    
    x_coords = triangles[:, :, 0]
    y_coords = triangles[:, :, 1]
    
    all_left = np.all(x_coords < x_min, axis=1)
    all_right = np.all(x_coords > x_max, axis=1)
    all_below = np.all(y_coords < y_min, axis=1)
    all_above = np.all(y_coords > y_max, axis=1)
    
    # A triangle is outside if it's completely to the left, right, below, or above
    outside = all_left | all_right | all_below | all_above
    
    return outside


def uv_project_from_view(ob, cam, uvm=0, clip_behind=True, view_clip=True, clip_range=1.0):
    
    print(f"Starting uv map for {cam.name}")
    co_uv, z_data, loops = co_to_uv(ob, cam)
    tridex = get_tridex_3(ob)

    clip_range = -clip_range
    move = np.array([-0.01, -0.01])
    # ---------------
    if loops.shape != tridex.ravel().shape:
        counts = np.array([len(p.vertices) for p in ob.data.polygons])
        uni = np.unique(counts)

        face_vert_counts = counts
        total_vert_count = np.sum(face_vert_counts)
        index_array = np.arange(total_vert_count)
        face_start_indices = np.zeros(len(face_vert_counts) + 1, dtype=int)
        face_start_indices[1:] = np.cumsum(face_vert_counts)
        
        grouped_indices = [index_array[face_start_indices[i]:face_start_indices[i+1]] 
                           for i in range(len(face_vert_counts))]
        uv_idxer = np.array(grouped_indices, dtype=object)

        if clip_behind:
            behind_cam = z_data[loops] > clip_range

        for n in uni:
            booler = counts == n
            uv_idx = np.array(uv_idxer[booler].tolist(), dtype=np.int32)
            if clip_behind:

                tri_bool = behind_cam[uv_idx]
                bad_tris = np.all(tri_bool, axis=1)
                tri_bool[:] = False
                tri_bool[bad_tris] = True
            else:
                tri_bool = np.zeros(uv_idx.ravel().shape[0], dtype=bool)
                tri_bool.shape = uv_idx.shape
                
            if view_clip:    
                outside = exclude_xy(co_uv[uv_idx], square_corner=(0.0, 0.0), square_size=1.0)
                tri_bool[outside] = True
            co_uv[uv_idx[tri_bool]] = move
        set_uv(ob, co_uv, uvm)
        return

    loops.shape = tridex.shape
    if np.all(loops == tridex):
        if clip_behind:
            behind_cam = z_data > clip_range
            tri_bool = behind_cam[tridex]
            bad_tris = np.all(tri_bool, axis=1)
            tri_bool[:] = False
            tri_bool[bad_tris] = True
        else:    
            behind_cam = np.zeros(z_data.shape[0], dtype=bool)
            tri_bool = behind_cam[tridex]
        
        co_uv.shape = (tridex.shape[0], tridex.shape[1], 2)
        if view_clip:    
            outside = exclude_xy(co_uv, square_corner=(0.0, 0.0), square_size=1.0)
            tri_bool[outside] = True
        co_uv[tri_bool] = move
    
    set_uv(ob, co_uv, uvm)


def copy_matrix():
    ob = bpy.context.object
    M = ob.matrix_world
    sel = [ob for ob in bpy.data.objects if ob.select_get()]
    for obs in sel:
        if obs != ob:
            obs.matrix_world = M
    

#=======================#
# IMAGE TOOLS ----------#
#=======================#
def get_pixels(img, include_size=False):
    pc = len(img.pixels)
    arr = np.zeros(pc, dtype=np.float32)
    img.pixels.foreach_get(arr)
    
    if include_size:
        h, w = img.size
        arr.shape = (h, w, 4)
        return arr

    return arr    
        

def set_pixels(img, arr):
    img.pixels.foreach_set(arr.ravel())


def quantize_to_nes_style(rgba_array, dithering=False):
    """
    Convert an RGBA numpy array (float 0-1) to a NES-style 8-bit look
    
    Parameters:
    -----------
    rgba_array : ndarray
        Input RGBA image as float32 with values between 0-1
    dithering : bool
        Whether to apply dithering for a more authentic retro look
        
    Returns:
    --------
    ndarray
        Quantized image with NES-style colors
    """
    # Make a copy to avoid modifying the original
    img = rgba_array.copy()
    
    # Convert from float [0-1] to integers [0-255]
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Define an NES-inspired palette (RGB values)
    # This is a simplified approximation of NES colors
    nes_palette = np.array([
        # Blacks/Grays/Whites
        [0, 0, 0], [84, 84, 84], [152, 152, 152], [236, 236, 236],
        # Blues
        [0, 0, 136], [0, 0, 204], [68, 40, 188], [120, 88, 248],
        # Greens
        [0, 120, 0], [0, 168, 0], [0, 168, 68], [0, 136, 136],
        # Reds/Browns
        [168, 0, 0], [204, 0, 0], [168, 16, 0], [136, 20, 0],
        # Oranges/Yellows
        [168, 84, 0], [236, 88, 0], [252, 152, 56], [252, 216, 84],
        # Purples/Pinks
        [168, 0, 168], [216, 0, 204], [248, 120, 248], [252, 176, 248]
    ], dtype=np.uint8)
    
    # Create KDTree for efficient nearest color matching
    tree = KDTree(nes_palette)
    
    # Reshape the image for processing
    original_shape = img_uint8.shape
    pixels = img_uint8[..., :3].reshape(-1, 3)
    alpha = None
    
    if original_shape[-1] == 4:  # If there's an alpha channel
        alpha = img_uint8[..., 3].reshape(-1)
    
    # Apply dithering if requested
    if dithering:
        # Simple Floyd-Steinberg dithering
        height, width = original_shape[:2]
        pixels_2d = pixels.reshape(height, width, 3)
        
        for y in range(height):
            for x in range(width):
                old_pixel = pixels_2d[y, x].copy().astype(np.float32)
                # Find closest color in palette
                _, idx = tree.query(old_pixel)
                new_pixel = nes_palette[idx]
                pixels_2d[y, x] = new_pixel
                
                # Calculate quantization error
                quant_error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x < width - 1:
                    pixels_2d[y, x+1] = np.clip(pixels_2d[y, x+1] + quant_error * 7/16, 0, 255)
                if y < height - 1:
                    if x > 0:
                        pixels_2d[y+1, x-1] = np.clip(pixels_2d[y+1, x-1] + quant_error * 3/16, 0, 255)
                    pixels_2d[y+1, x] = np.clip(pixels_2d[y+1, x] + quant_error * 5/16, 0, 255)
                    if x < width - 1:
                        pixels_2d[y+1, x+1] = np.clip(pixels_2d[y+1, x+1] + quant_error * 1/16, 0, 255)
        
        pixels = pixels_2d.reshape(-1, 3)
    
    # If not dithering, just map each pixel to the nearest palette color
    else:
        # Find the nearest color in the palette for each pixel
        _, indices = tree.query(pixels)
        pixels = nes_palette[indices]
    
    # Reshape back to original shape
    quantized = pixels.reshape(original_shape[:2] + (3,))
    
    # Add alpha channel back if it existed
    if alpha is not None:
        alpha = alpha.reshape(original_shape[:2] + (1,))
        quantized = np.concatenate([quantized, alpha], axis=2)
    
    # Convert back to float [0-1]
    quantized = quantized.astype(np.float32) / 255.0
    
    return quantized

    # use example:
    img = bpy.data.images['low res.jpeg']
    pix = get_pixels(img, include_size=True)
    qarr = quantize_to_nes_style(pix, dithering=False)
    print(qarr)
    set_pixels(img, qarr.ravel())


def merge_images(masks, img, overlays, final):
    """This would work if the uv map was
    the same for all the images..."""
    new_pix = np.ones_like(img)
    img_pix = get_pixels(img)

    for i in range(len(masks)):
        mask = masks[i]
        overlay = overlays[i]
        
        mask_pix = get_pixels(mask)
        alpha = mask_pix[:, 3]
        
        over_pix = get_pixels(overlay)
        booler = mask_pix != 0.0
        
        imgp = img_pix[:, 3] * (1 - alpha[:, None])
        over_pix[:3] *= alpha[:, None]
        mix = imgp + over_pix
        new_pix[booler] = mix[booler]
    
    new_pix[:, 3] = 1.0
    set_pixels(final, new_pix)


#=======================#
# POINT CLOUD ----------#
#=======================#
def o3d_sample_with_attributes(co, colors=None, voxel_size=0.01, return_indices=False):
    """Creates a uniform sample of points based on the voxel size
    and preserves point attributes like colors. Optionally returns
    indices of selected points from the original point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(co)
    
    # Add colors if provided
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Track original indices
    if return_indices:
        # Create a new attribute to store original indices
        indices = np.arange(len(co)).astype(np.float64)
        # Open3D requires attributes to be float64
        pcd.point_attributes = {'original_idx': o3d.utility.Vector1dVector(indices)}
    
    # Perform voxel downsampling
    uniform_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Get the downsampled points
    uniform_points = np.asarray(uniform_pcd.points)
    
    result = [uniform_points]
    
    # Get the downsampled colors if they were provided
    if colors is not None:
        uniform_colors = np.asarray(uniform_pcd.colors)
        result.append(uniform_colors)
    
    # Get the indices of selected points if requested
    if return_indices:
        # Extract the indices and convert back to integers
        selected_indices = np.asarray(uniform_pcd.point_attributes['original_idx']).astype(int)
        result.append(selected_indices)
    
    return tuple(result) if len(result) > 1 else uniform_points


def o3d_sample(co, voxel_size=0.01):
    """Creates a somewhat uniform sample of points
    based on the voxel size. Good for reducing the
    points in a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(co)
    uniform_pcd = pcd.voxel_down_sample(voxel_size)
    uniform_points = np.asarray(uniform_pcd.points)
    return uniform_points


def nearest_neighbor(points, co, k=1):
    """Find nearest neighbor in co for
    each point in points. Return index"""
    tree = KDTree(co)
    idxer = []
    for p in points:
        distances, indices = tree.query(p, k=k)
        idxer += [indices]
        
    return idxer


def sample_delete(ob=None, selected=[], box=None, distribute=False, voxel_size=0.02):
    
    if ob is None:
        ob = bpy.context.object
    if len(selected) == 0:
        sobs = [sob for sob in bpy.context.selected_objects if ob.type=="MESH"]
    
    v_counts = []
    for obj in sobs:
        v_counts += [len(obj.data.vertices)]    

    total_vert_count = np.sum(v_counts)
    total_co = np.empty((total_vert_count, 3), dtype=np.float32)
    
    if box is not None:
        local_co = np.empty((total_vert_count, 3), dtype=np.float32)

    total_colors = np.ones((total_vert_count, 4), dtype=np.float32)

    count = 0
    for i in range(len(sobs)):
        ob = sobs[i]
        vc = v_counts[i]
        count_plus = count + vc        
        co = get_co(ob)

        #if box is None:
            #lco = apply_transforms(ob, co, update_view=False)
        if box is not None:
            lco = co_to_matrix_space(co, ob, box)
            local_co[count:count_plus] = lco
        
        wco = apply_transforms(ob, co, update_view=False)
        total_co[count:count_plus] = wco
        
        color = get_color_attributes(ob, color=0)
        total_colors[count:count_plus] = color

        count = count_plus
        
    bco = get_co(box)
    bmin, bmax = get_bounds(bco)                
    in_box = co_in_bounds(local_co, bmin, bmax)
    
    box_co = total_co[in_box]
    sample_mesh = bpy.data.meshes.new("sampled_mesh")
    
    if distribute:
        sample = o3d_sample(box_co, voxel_size)
        vidx = np.arange(total_co.shape[0])[in_box]
        nnidx = nearest_neighbor(sample, box_co, k=1)
        nn = vidx[nnidx]
        
        sample_mesh.vertices.add(sample.shape[0])
        sample_mesh.vertices.foreach_set('co', sample.ravel())
        
        new_ob = bpy.data.objects.new("Distributed Points", sample_mesh)
        bpy.context.collection.objects.link(new_ob)
        set_color_attributes(new_ob, total_colors[nn].ravel(), color=0)
    
    else:
        sample_mesh.vertices.add(box_co.shape[0])
        new_ob = bpy.data.objects.new("Distributed Points", sample_mesh)
        bpy.context.collection.objects.link(new_ob)
        sample_mesh.vertices.foreach_set('co', box_co.ravel())
        set_color_attributes(new_ob, total_colors[in_box].ravel(), color=0)            


#box=bpy.data.objects['box']
#sample_delete(ob=None, selected=[], box=box, distribute=False, voxel_size=0.01)


#=======================#
# LATTICE TOOLS --------#
#=======================#
def get_selected(lat):
    vc = len(lat.data.points)
    sel = np.zeros(vc, dtype=bool)
    lat.data.points.foreach_get('select', sel)
    return sel
    

def get_neighbors(idx, res_u, res_v, res_w):
    """Grok smooth get neighbors"""
    u = idx % res_u
    v = (idx // res_u) % res_v
    w = idx // (res_u * res_v)
    neighbors = []
    
    # Check 6 possible neighbors (along U, V, W axes)
    for du, dv, dw in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
        nu, nv, nw = u + du, v + dv, w + dw
        if 0 <= nu < res_u and 0 <= nv < res_v and 0 <= nw < res_w:
            neighbor_idx = nu + nv * res_u + nw * res_u * res_v
            neighbors.append(neighbor_idx)
    return neighbors


def grok_smooth(lattice, iters=1, grow_steps=10, strength=1.0, base_key="PC Smooth Base"):
    """Smooth lattice coordinates using neighbor averages with falloff."""
    grow_steps += 1
    bpl = lattice.data
    points = bpl.points
    vc = len(points)

    # Get base coordinates
    active_key = lattice.active_shape_key
    keys = lattice.data.shape_keys
    if active_key:
        coords = np.empty((vc, 3))
        keys.key_blocks[base_key].data.foreach_get('co', coords.ravel())
    else:
        coords = np.array([p.co_deform for p in points])  # Deformed coordinates

    # Lattice dimensions
    res_u, res_v, res_w = bpl.points_u, bpl.points_v, bpl.points_w
    total_points = res_u * res_v * res_w

    # Selection and falloff computation
    selected = get_selected(lattice)
    falloff = np.zeros(total_points, dtype=np.float32)
    grow_selected = selected.copy()
    vidx = np.arange(total_points)
    v_sel = vidx[selected]
    multiplier = (1.0 / grow_steps) * strength
    count = grow_steps

    # Precompute neighbors for growing selection
    all_neighs = {}
    for v in vidx:
        u = v % res_u
        v_coord = (v // res_u) % res_v
        w = v // (res_u * res_v)
        neighbors = []
        for du, dv, dw in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nu, nv, nw = u + du, v_coord + dv, w + dw
            if 0 <= nu < res_u and 0 <= nv < res_v and 0 <= nw < res_w:
                neighbors.append(nu + nv * res_u + nw * res_u * res_v)
        all_neighs[v] = neighbors

    # Grow selection and compute falloff
    for i in range(grow_steps):
        if len(v_sel) == 0:
            break
        weight = multiplier * count
        falloff[v_sel] = weight
        grow_selected[v_sel] = True
        nn = list({n for vid in v_sel for n in all_neighs[vid]})
        v_sel = np.array(nn)[~grow_selected[nn]]
        count -= 1

    # Reshape to 3D grid
    coords_3d = coords.reshape(res_w, res_v, res_u, 3)
    falloff_3d = falloff.reshape(res_w, res_v, res_u)

    # Compute mean coordinates of neighbors
    sum_coords = np.zeros_like(coords_3d)
    count = np.zeros((res_w, res_v, res_u), dtype=int)

    # U direction
    sum_coords[:, :, :-1, :] += coords_3d[:, :, 1:, :]
    count[:, :, :-1] += 1
    sum_coords[:, :, 1:, :] += coords_3d[:, :, :-1, :]
    count[:, :, 1:] += 1

    # V direction
    sum_coords[:, :-1, :, :] += coords_3d[:, 1:, :, :]
    count[:, :-1, :] += 1
    sum_coords[:, 1:, :, :] += coords_3d[:, :-1, :, :]
    count[:, 1:, :] += 1

    # W direction
    sum_coords[:-1, :, :, :] += coords_3d[1:, :, :, :]
    count[:-1, :, :] += 1
    sum_coords[1:, :, :, :] += coords_3d[:-1, :, :, :]
    count[1:, :, :] += 1

    # Compute mean coordinates, avoiding division by zero
    mean_coords = np.divide(sum_coords, count[..., None], out=coords_3d.copy(), where=count[..., None] != 0)
    #return mean_coords
    # Compute final coordinates directly
    w = (1 - falloff_3d) ** iters
    new_coords_3d = mean_coords + w[..., None] * (coords_3d - mean_coords)
    new_coords = new_coords_3d.reshape(total_points, 3)

    return new_coords
    

def copy_key_data(l_data, axis=[0, 1, 2], copy_from="base_co", value=1.0):
    """Copy data from a shape key to the active
    shape key.
    copy_from: The shape key name or "base_co"
    axis: a list or array object including
    which axes to use: [0, 1, 2]
    value: How far to move towards the target data."""    
    
    ob = bpy.context.object
    active_key = ob.active_shape_key
    sel = l_data["selected"]
    co = l_data[active_key.name][sel][:, axis]
    copy_co = l_data[copy_from][sel][:, axis]
    
    selco = l_data[active_key.name][sel]
    
    dif = copy_co - co
    new = co + (dif * value)
    selco[:, axis] = new
    l_data[active_key.name][sel] = selco
    
    
    active_key.data.foreach_set('co', l_data[active_key.name].ravel())
    if ob.type=="MESH":
        ob.data.update()


def get_lattice_data(axis=[0, 1, 2], copy_from="base_co", value=1.0):
    ob = bpy.context.object
    if ob.type == "LATTICE":
        vc = len(ob.data.points)
        lattice = True
    else:
        vc = len(ob.data.vertices)
        lattice = False
    
    keys = ob.data.shape_keys
    l_data = {}

    mode = ob.mode
    edit = False
    if mode == "EDIT":
        edit = True
        bpy.ops.object.mode_set()
        #ob.data.points.foreach_get('co_deform', lco.ravel()) # get edit mode co for lattice, doesn't work for shape keys

    selected = np.zeros(vc, dtype=bool)
    lco = np.empty((vc, 3))
    if lattice:
        ob.data.points.foreach_get('select', selected)
        ob.data.points.foreach_get('co', lco.ravel())
    else:
        ob.data.vertices.foreach_get('select', selected)
        ob.data.vertices.foreach_get('co', lco.ravel())

    l_data["selected"] = selected
    l_data["base_co"] = lco
        
    if keys:
        for kb in keys.key_blocks:
            co = np.empty((vc, 3))
            kb.data.foreach_get('co', co.ravel())
            l_data[kb.name] = co

    copy_key_data(l_data, axis, copy_from, value)

    if edit:
        bpy.ops.object.mode_set(mode="EDIT")
        
    return l_data
            

def edges_from_matrix(matrix):
    ed = np.empty((0,2), dtype=np.int32)
    
    # get all cubes...
    shape = matrix.shape
    s1, s2, s3 = shape
    total_cubes = (shape[0] - 1) * (shape[1] - 1) * (shape[2] - 1)
    print(shape, "shape?")
    print(total_cubes, "total?")
    cubes = np.zeros((total_cubes, 8), dtype=np.int32)
    cubes[:] = -1

    cubes[:, 0] = matrix[:-1, :-1, :-1].ravel()
    cubes[:, 1] = matrix[:-1, :-1, 1:].ravel()
    cubes[:, 2] = matrix[:-1, 1:, :-1].ravel()
    cubes[:, 3] = matrix[:-1, 1:, 1:].ravel()

    cubes[:, 4] = matrix[1:, :-1, :-1].ravel()
    cubes[:, 5] = matrix[1:, :-1, 1:].ravel()
    cubes[:, 6] = matrix[1:, 1:, :-1].ravel()
    cubes[:, 7] = matrix[1:, 1:, 1:].ravel()
    
    eds = np.empty((0, 2), dtype=np.int32)
    for c in cubes:
        
        ed = np.empty((16, 2), dtype=np.int32)        
        ed[:, 0] = c[[0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 2, 3, 0, 1, 2, 3]]
        ed[:, 1] = c[[1, 3, 2, 0, 5, 7, 6, 4, 4, 5, 6, 7, 7, 6, 5, 4]]
        
        eds = np.append(eds, ed, axis=0)
        
    return eds    
    

def get_based_lengths(ob, ed):
    verts = ob.data.points
    vc = len(verts)
    co = np.empty((vc, 3), dtype=np.float32)
    verts.foreach_get('co', co.ravel())
    vecs = co[ed[:,1]] - co[ed[:, 0]]
    # based lengths (as opposed to lengths who identify as different than their physical measurement)
    based_lengths = measure_vecs(vecs)
    return based_lengths
    

def grid_smooth_er(co, eidx, sel, based_lengths):

    lat = bpy.context.object
    sel_edges = np.any(sel[eidx], axis=1)
    edges_I_care_about = eidx[sel_edges]
    lens_I_care_about = based_lengths[sel_edges]
    difs = co[edges_I_care_about[:, 1]] - co[edges_I_care_about[:, 0]]
    dists = measure_vecs(difs)
    u_vecs = difs / dists[:, None]
    
    vidx = np.arange(sel.shape[0])
    sel_v = vidx[sel]

    ec = edges_I_care_about.shape[0]
    flip = np.ones((ec, 2), dtype=np.float32)
    flip[:, 0] = -1
    
    move = np.zeros_like(co)
    for v in sel_v:
        in_edges = edges_I_care_about == v
        flipper = flip[in_edges]
        
        hits = np.any(in_edges, axis=1)
        otvs = edges_I_care_about[hits][~in_edges[hits]]
        lens = lens_I_care_about[hits]
        pos = co[otvs] + u_vecs[hits] * (flipper * lens)[:, None]
        meaner = np.mean(pos, axis=0)
        move[v] = meaner - co[v]
    
    sco = co + move * lat.PC_props.lattice_smooth
    return sco


def get_mean_data(lat):
    
    verts = lat.data.points
    u, v, w = lat.data.points_u, lat.data.points_v, lat.data.points_w
    vc = len(verts)

    co = np.empty((vc, 3))
    lat.data.points.foreach_get('co', co.ravel())
    

    current_shape_co = np.empty((vc, 3))
    smooth_base_co = np.empty((vc, 3))
    lat.active_shape_key.data.foreach_get("co", current_shape_co.ravel())
    lat.data.shape_keys.key_blocks['PC Smooth Base'].data.foreach_get("co", smooth_base_co.ravel())
    
    matrix = np.arange(vc)
    matrix.shape = (u, v, w)

    print(matrix, "working?")
    data = {"matrix": matrix}
    return data

    
def lattice_smooth_mean(lat, prop):
    mode = lat.mode    
    edit = False
    if mode == "EDIT":
        edit = True
        bpy.ops.object.mode_set()
    
    strength = lat.PC_props.lattice_smooth
    grow_steps = lat.PC_props.lattice_smooth_grow_steps
    iters = lat.PC_props.lattice_smooth_iters

    sco = grok_smooth(lat, iters, grow_steps, strength, base_key="PC Smooth Base")

    lat.active_shape_key.data.foreach_set("co", sco.ravel())
    lat.data.points.update()
    
    if edit:
        bpy.ops.object.mode_set(mode="EDIT")    
    
    

def lattice_smooth(lat, prop, count=1):

    mode = lat.mode    
    edit = False
    
    if mode == "EDIT":
        edit = True
        bpy.ops.object.mode_set()

    
    verts = lat.data.points
    u, v, w = lat.data.points_u, lat.data.points_v, lat.data.points_w
    vc = len(verts)

    co = np.empty((vc, 3))
    lat.data.points.foreach_get('co', co.ravel())
    
    current_shape_co = np.empty((vc, 3))
    smooth_base_co = np.empty((vc, 3))
    lat.active_shape_key.data.foreach_get("co", current_shape_co.ravel())
    lat.data.shape_keys.key_blocks['PC Smooth Base'].data.foreach_get("co", smooth_base_co.ravel())
    
    matrix = np.arange(vc)
    matrix.shape = (u, v, w)

    eidx = edges_from_matrix(matrix)
    based_lengths = get_based_lengths(lat, eidx)
    sel = np.zeros(vc, dtype=bool)
    sel[eidx[count]] = True
    
    verts.foreach_get("select", sel)
        
    sco = grid_smooth_er(smooth_base_co, eidx, sel, based_lengths)
    for i in range(3):
        sco = grid_smooth_er(sco, eidx, sel, based_lengths)
    lat.active_shape_key.data.foreach_set("co", sco.ravel())
    lat.data.points.update()
            
    if edit:
        bpy.ops.object.mode_set(mode="EDIT")    




# magic dot power in the multiverse
# magic dot power in the multiverse
# magic dot power in the multiverse
def dots(a,b):
    #N x 3 - N x 3
    x = np.einsum('ij,ij->i', a, b)
    #3 - N x 3
    y = np.einsum('j,ij->i', a, b) # more simply b @ n
    #N x 3 - N x N x 3
    z = np.einsum('ij,ikj->ik', a, b)    
    #N x N x 3 - N x N x 3    
    w = np.einsum('ijk,ijk->ij', a, b)
    #N x 2 x 3 - N x 2 x 3
    a = np.einsum('ijk,ijk->ij', a, b)    
    
    # Nx3 - Nx3 broadcasted to NxN
    np.einsum('ij, kj -> ik', vecs, vecs)
    
    #N x N x 3 - 3 x 3
    z = np.einsum('ikj,ij->ik', ori_vecs, ori)
    
    #N x 2 x 3 - N x 3
    np.einsum('ij, ikj->ik', axis_vecs, po_vecs)
    
    #N x 3 x 3 - N x 3
    np.einsum('ijk,ik->ij', a, b)
    #N x 3 x 3 - N x 3 with transpose
    np.einsum('ikj,ik->ij', a, b)
    
    # copy and rebuild matrix with N x 3 x 3 and Nx3 coords
    scalers = np.einsum('vij,vj->vi', vert_matrix, local_co) # don't forget to use the same origin to get vectors
    new_locs = np.einsum('vi,vij->vj', scalers, vert_matrix)
    
    #mismatched N x 3 - N2 x 3 with broadcasting so that the end result is tiled
    mismatched = np.einsum('ij,i...j->...i', a, np.expand_dims(b, axis=0))    
    # 4,3,3 - 4,2,3 with broadcasting    
    mismatched_2 = np.einsum('ijk,ij...k->i...j', a, np.expand_dims(b, axis=1))
    return x,y,z,w,a, mismatched, mismatched_2


# list of mods other than deforming mods
('DATA_TRANSFER',
'MESH_CACHE', 
'MESH_SEQUENCE_CACHE',
'NORMAL_EDIT',
'WEIGHTED_NORMAL',
'UV_PROJECT',
'UV_WARP',
'GREASE_PENCIL_COLOR',
'GREASE_PENCIL_TINT',
'GREASE_PENCIL_OPACITY',
'ARRAY',
'BEVEL',
'BOOLEAN',
'BUILD',
'DECIMATE',
'EDGE_SPLIT', 
'NODES', 'MASK', 'MIRROR',
'MESH_TO_VOLUME',
'MULTIRES', 'REMESH', 
'SCREW', 'SKIN', 
'SOLIDIFY', 
'SUBSURF', 
'TRIANGULATE', 
'VOLUME_TO_MESH',
'WELD', 'WIREFRAME', 
'GREASE_PENCIL_SUBDIV', 
'GREASE_PENCIL_MIRROR', 
'GREASE_PENCIL_NOISE',
'GREASE_PENCIL_OFFSET',
'GREASE_PENCIL_SMOOTH',
'GREASE_PENCIL_THICKNESS',
'COLLISION',
'DYNAMIC_PAINT',
'EXPLODE',
'FLUID',
'OCEAN',
'PARTICLE_INSTANCE',
'PARTICLE_SYSTEM', 
'SURFACE')


#box_around_selected()


if False: # cleaning stuff
    print("new check")
    for ob in bpy.data.objects:
        if ob.type == "MESH":    
            for m in ob.data.materials:
                if m.name.startswith("23"):
                    bpy.context.scene.collection.objects.link(ob)
                    print(m.name, "mat name")
                    print(ob.name, "ob name")


def form_quads_and_tris(row_idx, col_idx):
    """
    Group point indices into quads and triangles based on row and column indices.
    
    Parameters:
    - row_idx: NumPy array of row indices for each point
    - col_idx: NumPy array of column indices for each point
    
    Returns:
    - quads: Nx4 NumPy array where each row contains indices of four points forming a quad
    - tris: Mx3 NumPy array where each row contains indices of three points forming a triangle
    """
    # Find the range of row and column indices
    r_min, r_max = np.min(row_idx), np.max(row_idx)
    c_min, c_max = np.min(col_idx), np.max(col_idx)
    
    # Initialize a grid with -1
    grid = np.full((r_max - r_min + 1, c_max - c_min + 1), -1, dtype=np.int64)
    
    # Populate the grid with point indices
    grid[row_idx - r_min, col_idx - c_min] = np.arange(len(row_idx))
    
    # Create a mask for existing points
    mask = (grid != -1)
    
    # Identify positions where all four points of a quad exist
    quad_mask = (
        mask[:-1, :-1] & mask[:-1, 1:] & 
        mask[1:, :-1] & mask[1:, 1:]
    )
    
    # Get indices where quads can be formed
    i_indices, j_indices = np.where(quad_mask)
    
    # Initialize output array for quads
    quads = np.zeros((len(i_indices), 4), dtype=np.int64)
    
    # Fill the quads array with point indices
    quads[:, 0] = grid[i_indices, j_indices]              # (i, j)
    quads[:, 1] = grid[i_indices, j_indices + 1]          # (i, j+1)
    quads[:, 2] = grid[i_indices + 1, j_indices + 1]      # (i+1, j+1)
    quads[:, 3] = grid[i_indices + 1, j_indices]          # (i+1, j)
    
    # Define masks for triangles where exactly three points are present
    missing_top_left = ~mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    missing_top_right = mask[:-1, :-1] & ~mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    missing_bottom_left = mask[:-1, :-1] & mask[:-1, 1:] & ~mask[1:, :-1] & mask[1:, 1:]
    missing_bottom_right = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & ~mask[1:, 1:]
    
    # Find indices for each triangle case
    i_mtl, j_mtl = np.where(missing_top_left)
    i_mtr, j_mtr = np.where(missing_top_right)
    i_mbl, j_mbl = np.where(missing_bottom_left)
    i_mbr, j_mbr = np.where(missing_bottom_right)
    
    # Collect triangles for each case
    tris = []
    if len(i_mtl) > 0:
        tris.append(np.column_stack([grid[i_mtl, j_mtl + 1], grid[i_mtl + 1, j_mtl + 1], grid[i_mtl + 1, j_mtl]]))
    if len(i_mtr) > 0:
        tris.append(np.column_stack([grid[i_mtr, j_mtr], grid[i_mtr + 1, j_mtr + 1], grid[i_mtr + 1, j_mtr]]))
        #tris.append(np.column_stack([grid[i_mtr, j_mtr], grid[i_mtr + 1, j_mtr], grid[i_mtr + 1, j_mtr + 1]]))
    if len(i_mbl) > 0:
        tris.append(np.column_stack([grid[i_mbl, j_mbl], grid[i_mbl, j_mbl + 1], grid[i_mbl + 1, j_mbl + 1]]))
    if len(i_mbr) > 0:
        tris.append(np.column_stack([grid[i_mbr, j_mbr], grid[i_mbr, j_mbr + 1], grid[i_mbr + 1, j_mbr]]))
    
    # Combine all triangles into a single array
    if tris:
        tris = np.vstack(tris)
    else:
        tris = np.empty((0, 3), dtype=np.int64)
    
    return quads, tris


def filter_by_max_edge_length(points, faces, min_div):
    """
    Filter faces (triangles or quadrilaterals) to remove those with any edge longer than max_edge_length.
    
    Parameters:
    - points: NumPy array of shape (P, 3), coordinates of the points in 3D space
    - faces: NumPy array of shape (N, K), indices of points forming the faces (K=3 for tris, K=4 for quads)
    - max_edge_length: float, the maximum allowed edge length
    
    Returns:
    - filtered_faces: NumPy array of shape (M, K), the faces that satisfy the edge length condition
    """
    if faces.size == 0:
        return faces
    # Get coordinates of points for each face
    points_faces = points[faces]  # Shape (N, K, 3)
    # Compute edge vectors by shifting points cyclically and subtracting
    edges = np.roll(points_faces, -1, axis=1) - points_faces  # Shape (N, K, 3)
    # Calculate edge lengths using Euclidean norm
    edge_lengths = np.linalg.norm(edges, axis=2)  # Shape (N, K)
    # Find the maximum edge length for each face
    max_edges = np.max(edge_lengths, axis=1)  # Shape (N,)
    min_edges = np.min(edge_lengths, axis=1)  # Shape (N,)
    
    div = min_edges / max_edges
    
    # Keep faces where all edges are within the limit
    valid = div > 0.25
    return faces[valid]


def form_quads(row_idx, col_idx):
    """
    Group point indices into quads based on row and column indices.
    
    Parameters:
    - row_idx: NumPy array of row indices for each point
    - col_idx: NumPy array of column indices for each point
    
    Returns:
    - quads: Nx4 NumPy array where each row contains indices of four points forming a quad
    """
    # Step 1: Find the range of row and column indices
    r_min, r_max = np.min(row_idx), np.max(row_idx)
    c_min, c_max = np.min(col_idx), np.max(col_idx)
    
    # Step 2: Initialize a grid with -1
    grid = np.full((r_max - r_min + 1, c_max - c_min + 1), -1, dtype=np.int64)
    
    # Step 3: Populate the grid with point indices
    grid[row_idx - r_min, col_idx - c_min] = np.arange(len(row_idx))
    
    # Step 4: Create a mask for existing points
    mask = (grid != -1)
    
    # Step 5: Identify positions where all four points of a quad exist
    quad_mask = (
        mask[:-1, :-1] & mask[:-1, 1:] & 
        mask[1:, :-1] & mask[1:, 1:]
    )
    
    # Step 6: Get indices where quads can be formed
    i_indices, j_indices = np.where(quad_mask)
    
    # Step 7: Initialize output array for quads
    quads = np.zeros((len(i_indices), 4), dtype=np.int64)
    
    # Step 8: Fill the quads array with point indices
    quads[:, 0] = grid[i_indices, j_indices]              # (i, j)
    quads[:, 1] = grid[i_indices, j_indices + 1]          # (i, j+1)
    quads[:, 2] = grid[i_indices + 1, j_indices + 1]      # (i+1, j+1)
    quads[:, 3] = grid[i_indices + 1, j_indices]          # (i+1, j)
    
    return quads


def measure_angle_at_each_vert(co):
    """Assumes co is a looped polyline
    with verts in order. Returns the dot
    product of the vecs pointing away
    from each other at each vert """
    
    v1 = np.roll(co, 1, axis=0) - co
    v2 = np.roll(co, -1, axis=0) - co
    
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, None]
    uv2 = v2 / np.sqrt(rs_dots)[:, None]
    
    angles = np.einsum('ij, ij->i', uv1, uv2)
    return angles


def form_all_tris(row_idx, col_idx):
    """
    Generate a single array of triangles from grid points.
    - Splits quadrilaterals into two triangles where all four points exist.
    - Keeps triangles where exactly three points exist.
    - Ensures all triangles are oriented counter-clockwise.
    
    Parameters:
    - row_idx: NumPy array of row indices for each point
    - col_idx: NumPy array of column indices for each point
    
    Returns:
    - all_tris: Mx3 NumPy array, each row contains indices of three points forming a triangle
    """
    # Determine grid dimensions
    r_min, r_max = np.min(row_idx), np.max(row_idx)
    c_min, c_max = np.min(col_idx), np.max(col_idx)
    
    # Create grid initialized with -1
    grid = np.full((r_max - r_min + 1, c_max - c_min + 1), -1, dtype=np.int64)
    
    # Populate grid with point indices
    grid[row_idx - r_min, col_idx - c_min] = np.arange(len(row_idx))
    
    # Mask for existing points
    mask = (grid != -1)
    
    # Identify quads (all four points present)
    quad_mask = (
        mask[:-1, :-1] & mask[:-1, 1:] & 
        mask[1:, :-1] & mask[1:, 1:]
    )
    i_indices, j_indices = np.where(quad_mask)
    
    # List to collect all triangles
    all_tris = []
    
    # Split each quad into two triangles
    if len(i_indices) > 0:
        tri1 = np.column_stack([
            grid[i_indices, j_indices],           # (i,j)
            grid[i_indices, j_indices + 1],       # (i,j+1)
            grid[i_indices + 1, j_indices + 1]    # (i+1,j+1)
        ])
        tri2 = np.column_stack([
            grid[i_indices, j_indices],           # (i,j)
            grid[i_indices + 1, j_indices + 1],   # (i+1,j+1)
            grid[i_indices + 1, j_indices]        # (i+1,j)
        ])
        all_tris.extend([tri1, tri2])
    
    # Identify triangles (exactly three points present)
    missing_top_left = ~mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    missing_top_right = mask[:-1, :-1] & ~mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    missing_bottom_left = mask[:-1, :-1] & mask[:-1, 1:] & ~mask[1:, :-1] & mask[1:, 1:]
    missing_bottom_right = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & ~mask[1:, 1:]
    
    # Collect triangles based on missing vertex
    for missing_mask, tri_order in [
        (missing_top_left, lambda i, j: [grid[i, j+1], grid[i+1, j+1], grid[i+1, j]]),
        (missing_top_right, lambda i, j: [grid[i, j], grid[i+1, j+1], grid[i+1, j]]),
        (missing_bottom_left, lambda i, j: [grid[i, j], grid[i, j+1], grid[i+1, j+1]]),
        (missing_bottom_right, lambda i, j: [grid[i, j], grid[i, j+1], grid[i+1, j]])
    ]:
        i_m, j_m = np.where(missing_mask)
        if len(i_m) > 0:
            tris_m = np.column_stack(tri_order(i_m, j_m))
            all_tris.append(tris_m)
    
    # Combine all triangles into one array
    if all_tris:
        all_tris = np.vstack(all_tris)
    else:
        all_tris = np.empty((0, 3), dtype=np.int64)
    
    return all_tris


def compute_vert_norms(ob, tri_norms, tridex):
    """Starts with scan norms as default then replaces
    scan norms where possible with average normal
    based on shared triangles."""
    merge_norms = get_named_vector_attribute(ob, "scanNormals")
    #boolingtons = np.ones = len(ob.data.vertices, dtype=bool)
    merge_norms[tridex.ravel()] = 0.0
    print(merge_norms, "merge norms?")
    np.add.at(merge_norms, tridex.ravel(), np.repeat(tri_norms, 3, axis=0))
    norms = u_vecs(merge_norms)
    return norms


def mesh_from_rows_cols(ob, co=None):

    row_idx = get_named_int_attribute(ob, name="rowIndex")
    col_idx = get_named_int_attribute(ob, name="columnIndex")
    #quads = form_quads(row_idx, col_idx)
    #quads, tris = form_quads_and_tris(row_idx, col_idx)
    
    if co is None:
        co = get_co(ob)#[:test]
    
    separate_qt = False
    if separate_qt:
        tris = filter_by_max_edge_length(co, tris, min_div=0.025)
        quads = filter_by_max_edge_length(co, quads, min_div=0.025)
    
    # Quads and tris together:
    qt = form_all_tris(row_idx, col_idx)
    tri_norms = get_tri_normals(co[qt], normalize=False)

    merge_norms = compute_vert_norms(ob, tri_norms, qt)    
    set_named_vector_attribute(ob, merge_norms, name="surfaceNormals", type="FLOAT_VECTOR", domain="POINT")
    
    ob_from_py_data(co, faces=qt, edges=[], name="new mesh")
    return co, tris, merge_norms


if False:    
    mesh_from_rows_cols(bpy.context.object)

if False:
    ob = bpy.context.object
    #ob.modifiers['GeometryNodes'].node_group.nodes['flip_faces'].inputs['Object'].default_value = None
    #ob.data.update()
    hidden_object = attribute_object(ob, use_prox=True)
    ob.modifiers['GeometryNodes'].node_group.nodes['flip_faces'].inputs['Object'].default_value = hidden_object


if False:
    box = bpy.data.objects['clip']
    delete_in_box(box, [ob for ob in bpy.data.objects if ob.select_get()])


def flattened_loop_co(ob, vidx=None):
    """Uses bmesh and uv unwrapping to to get a flattened version
    of a polyline"""
    obm = get_bmesh(ob, refresh=True)
    if vidx is None:    
        vidx = get_ordered_loop(ob, edges=None, obm=obm)
        
    check_faces(ob, vidx)
    unwrap_object(ob)

    idx = ob.data.uv_layers.active_index
    uvc = uv_shape(ob, uvm=idx, skip=True)
    delete_faces(ob, obm=None, face_idx=[0], type=0)    

    return uvc, vidx

#check_faces(bpy.context.object)
#unwrap_object(bpy.context.object)