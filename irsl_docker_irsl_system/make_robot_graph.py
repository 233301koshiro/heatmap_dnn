## make random tree
import networkx as nx
import random
import copy
random.seed()
from networkx.drawing.nx_agraph import write_dot
## write_dot(G, "/tmp/hoge.dot")

exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())

_flatten = lambda x: [z for y in x for z in (_flatten(y) if type(y) is list else (y,))]

def _axis_to_rotation(axis, target_axis=coordinates.Y):
    cds = coordinates()
    if   type(axis) is str and axis == 'X':
        cds.rotate(-PI/2, coordinates.Z)
    elif type(axis) is str and axis == 'Y':
        pass## cds
    elif type(axis) is str and axis == 'Z':
        cds.rotate( PI/2, coordinates.X)
    elif type(axis) is str and axis == '-X':
        cds.rotate( PI/2, coordinates.Z)
    elif type(axis) is str and axis == '-Y':
        cds.rotate( PI, coordinates.Z)##
    elif type(axis) is str and axis == '-Z':
        cds.rotate(-PI/2, coordinates.X)
    else: ## axis is list
        ax = np.array(axis, dtype='float64')
        IC.normalizeVector(ax)
        cds = ru.axisAlignedCoords(ax, target_axis)
    ##
    return cds

##
## Geometry generator
## TODO: class method ?
##
class GeometryGenerator(object):
    def __init__(self, func_dict):
        self.func_dict = func_dict
    def generate(self, geomtype, *args, **kwargs):
        if geomtype in self.func_dict:
            return self.func_dict[geomtype](*args, **kwargs)
        else:
            return (None, None)
##
def _func_fix_pre(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    cds = joint_coords.copy()
    if color is None:
        color = [0, 0, 0.4]
    if axis is None:
        cds.translate(fv(0, -0.1*scale, 0))
    else:
        cds.translate(axis*-0.1*scale)
    geometry = {'primitive': 'box',
                'args': {'x': 0.2*scale, 'color': color },
                }
    return (cds, geometry)
def _func_fix_post(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    cds = joint_coords.copy()
    if color is None:
        color = [0, 0, 0.8]
    if axis is None:
        cds.translate(fv(0, 0.1*scale, 0))
    else:
        cds.translate(axis*0.1*scale)
    geometry = {'primitive': 'box',
                'args': {'x': 0.2*scale, 'color': color },
                }
    return (cds, geometry)
##
def _func_revolute_pre(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    height_  = 0.25 * scale
    radius_ = 0.10 * scale
    cds = joint_coords.copy()
    mv_ = fv(0, -height_/2, 0)
    trs = _axis_to_rotation(axis)
    trs.translate(mv_)
    cds.transform(trs)
    if color is None:
        color = [0.5, 0, 0]
    geometry = {'primitive': 'cylinder',
                'args': {'radius': radius_, 'height': height_, 'color': color },
                }
    return (cds, geometry)
def _func_revolute_post(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    height_ = 0.25 * scale
    radius_ = 0.10 * scale
    cds = joint_coords.copy()
    mv_ = fv(0,  height_/2, 0)
    trs = _axis_to_rotation(axis)
    trs.translate(mv_)
    cds.transform(trs)
    if color is None:
        color = [1.0, 0, 0]
    geometry = {'primitive': 'cylinder',
                'args': {'radius': radius_, 'height': height_, 'color': color },
                }
    return (cds, geometry)
##
def _func_revolute_yaw_pre(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    height_ = 0.05 * scale
    radius_ = 0.25 * scale
    cds = joint_coords.copy()
    mv_ = fv(0, -height_/2, 0)
    trs = _axis_to_rotation(axis)
    trs.translate(mv_)
    cds.transform(trs)
    if color is None:
        color = [0.5, 0, 0]
    geometry = {'primitive': 'cylinder',
                'args': {'radius': radius_, 'height': height_, 'color': color },
                }
    return (cds, geometry)
def _func_revolute_yaw_post(joint_coords, axis=None, scale=1.0, color=None, **kwargs):
    height_ = 0.05 * scale
    radius_ = 0.25 * scale
    cds = joint_coords.copy()
    mv_ = fv(0,  height_/2, 0)
    trs = _axis_to_rotation(axis)
    trs.translate(mv_)
    cds.transform(trs)
    if color is None:
        color = [1.0, 0, 0]
    geometry = {'primitive': 'cylinder',
                'args': {'radius': radius_, 'height': height_, 'color': color },
                }
    return (cds, geometry)
##
def _func_links(parent_coords, child_coords, scale=1.0, color=None, **kwargs): ## BOX type
    direction = child_coords.pos - parent_coords.pos
    len_ = np.linalg.norm(direction)
    if len_ < 1.e-5:
        return (None, None)
    cds = ru.axisAlignedCoords(direction, coordinates.Y)
    cds.pos = parent_coords.pos
    cds.translate(0.5*len_*coordinates.Y)
    geometry = {'primitive': 'box',
                'args': {'x': 0.2, 'y': len_, 'z': 0.2, 'color': [1, 1, 1] },
                }
    return (cds, geometry)
def _func_links_cylinder(parent_coords, child_coords, scale=1.0, color=None, **kwargs):
    direction = child_coords.pos - parent_coords.pos
    len_ = np.linalg.norm(direction)
    if len_ < 1.e-5:
        return (None, None)
    cds = ru.axisAlignedCoords(direction, coordinates.Y)
    cds.pos = parent_coords.pos
    cds.translate(0.5*len_*coordinates.Y)
    geometry = {'primitive': 'cylinder',
                'args': {'radius': 0.06*scale, 'height': len_, 'color': [0, 1, 0] },
                }
    return (cds, geometry)

class RobotTree(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 0: ## converted from networkx.DiGraph
            for n in self.nodes:
                node = self.nodes[n]
                if 'params' in node:
                    str_param = node['params']
                    if type(str_param) is str:
                        self.nodes[n]['params'] = eval(eval(str_param))
        self.root = None
        self.root_coords = coordinates()
        ### generator(*1*)
        self.generator = GeometryGenerator(
            {"prev_revolute":     _func_revolute_pre,
             "prev_revolute_yaw": _func_revolute_yaw_pre,
             "prev_linear":       _func_fix_pre,
             "prev_fixed":        _func_fix_pre,
             "prev_ball":         _func_fix_pre,
             "post_revolute":     _func_revolute_post,
             "post_revolute_yaw": _func_revolute_yaw_post,
             "post_linear":       _func_fix_post,
             "post_fixed":        _func_fix_post,
             "post_ball":         _func_fix_post,
             "between_links":     _func_links_cylinder,
             })
    ##
    ##
    ##
    def write_dot(self, fname, down_convert=True):
        if down_convert:
            self.down_convert_coords()
        write_dot(self, fname)

    @staticmethod
    def read_dot(fname, up_convert=True):
        res = RobotTree(nx.nx_pydot.read_dot(fname))
        if up_convert:
            res.up_convert_coords()
        return res
    ##
    ## methods: common tree
    ##
    def search_root(self):
        """
        """
        root=None
        for n in self.nodes:
            if len(list(self.predecessors(n))) == 0:
                if root is not None:
                    ## warning there are multiple root(graph)
                    pass
                root=n
        self.root = root
        return root

    def search_leafs(self):
        """
        """
        res = []
        for n in self.nodes:
            if len(list(self.successors(n))) == 0:
                res.append(n)
        return res

    def check_tree(self):
        """
        """
        ## num of parents is just one
        for node in self.nodes:
            pred = list(self.predecessors(node))
            succ = list(self.successors(node))
            if len(pred) > 1:
                ###
                return False
            if len(succ) == 0 and len(pred) == 0:
                ###
                return False
        return True

    def remove_edges_to_node(self, node, fix=True, new_node=None):
        """
        """
        pred = list(self.predecessors(node))
        succ = list(self.successors(node))
        for p in pred:
            self.remove_edge(p, node)
        for s in succ:
            self.remove_edge(node, s)
        if fix:
            if new_node is None:
                if len(pred) > 0:
                    for s in succ:
                        self.add_edge(pred[0], s)
            else:
                for p in pred:
                    self.add_edge(p, new_node)
                for s in succ:
                    self.add_edge(new_node, s)

    def find_far_edge(self, root, cycle):
        farlen = 0
        faredge = None
        for p, c in cycle:
            pst = nx.shortest_path(self, root, p)
            if len(pst) > farlen:
                farlen = len(pst)
                faredge = (p, c)
        return (farlen, faredge)

    def remove_cycle_edges(self, root):
        while True:
            cycle = self.find_cycle(self)
            if cycle is None:
                return
            farlen, faredge = self.find_far_edge(root, cycle)
            if farlen > 0:
                self.remove_edge(*faredge)

    def find_cycle(self):
        cycle = None
        try:
            cycle = nx.find_cycle(self)
        except Exception as e:
            pass
        return cycle

    def insert_node(self, parent, child, node, **kwargs):
        pass

    def change_root(self, new_root):
        """
        """
        if not new_root in self.nodes:
            raise Exception('node: {} does not exist'.format(new_root))
        old_root = self.search_root()
        res = nx.shortest_path(self, old_root, new_root)
        if len(res) < 1:
            raise Exception('no path from {} to {}'.format(old_root, new_root))
        prv = old_root
        for cur in res[1:]:
            self.remove_edge(prv, cur)
            self.add_edge(cur, prv)
            prv = cur
        return old_root

    def change_robot_root(self, new_root):
        old_root = self.change_root(new_root)
        params = self.nodes[old_root]['params']
        if 'joint' in params and 'type' in params['joint'] and params['joint']['type'] == 'root':
            pred = list(self.predecessors(old_root))
            if len(pred) != 1:
                raise Exception('{} has more than one predecessors. {}'.format(old_root, pred))
            self.remove_edge(pred[0], old_root)
            self.add_edge(old_root, new_root)
        else:
            raise Exception('old root({}) is not "Root"'.format(old_root))
    #
    # Data
    #
    def parse_list(self, lst, parent=None, add_root=True):
        """
        making graph(tree) from list-rep (inverse of make_list_from_tree)

        Argas:
            lst (list[ dict ]) :
            parent (dict, optional) : params
            add_root (boolean, default=True) :
        Retuns:
            RobotTree : Output

        """
        if len(lst) < 1:
            return
        if parent is None:
            if add_root or (type(lst[0]) is list)  or  RobotTree.check_type(lst[0]) == 1:
                parent={'node': 'Root', 'type': 'link', 'joint': {'type': 'root'}}
            else:
                parent=lst[0]
                lst=lst[1:]
        car=lst[0]
        cdr=lst[1:]
        if type(car) is list:
            self.parse_list(car, parent=parent)
            self.parse_list(cdr, parent=parent)
        else:
            self.add_node(str(parent['node']), params=copy.deepcopy(parent))
            self.add_node(str(car['node']), params=copy.deepcopy(car))
            self.add_edge(str(parent['node']), str(car['node']))
            self.parse_list(cdr, parent=car)

    @staticmethod
    def generate_from_list(lst, add_root=True):
        """
        making graph(tree) from list-rep (inverse of make_list_from_tree)

        Argas:
            lst (list[ dict ]) :
            add_root (boolean, default=True) :
        Retuns:
            RobotTree : Output

        """
        G = RobotTree()
        G.parse_list(lst, add_root=add_root)
        return G

    def make_list(self, start_node=None, use_params=True):
        """
        Dump tree data as a list
        """
        if start_node is None:
            start_node = self.search_root()
        succ = list(self.successors(start_node))
        if use_params:
            st_params = self.nodes[start_node]['params']
            st_params['node'] = start_node
            start_node = st_params
        if len(succ) > 0:
            child=succ[0]
        else:
            return [start_node]
        result = [start_node]
        if len(succ) > 1:
            siblings=succ[1:]
            for s in siblings:
                result.append(self.make_list(start_node=s, use_params=use_params))
        result += self.make_list(start_node=child, use_params=use_params)
        return result

    #
    # handling RobotTree
    #
    @staticmethod
    def _up_to_coordinates(node_map, force=False):
        if not force:
            if 'coords' in node_map and type(node_map['coords']) is coordinates:
                return
        params = node_map['params']
        if 'translation' in params or 'rotation' in params:
            node_map['coords'] = ru.make_coordinates(params)
            if 'translation' in params:
                del params['translation']
            if 'rotation' in params:
                del params['rotation']
        elif not 'coords' in node_map:
            node_map['coords'] = coordinates()
        elif node_map['coords'] is None:
            node_map['coords'] = coordinates()
        elif type(node_map['coords']) is dict:
            coords_map = node_map['coords']
            node_map['coords'] = ru.make_coordinates(coords_map)
        elif type(node_map['coords']) is str:
            coords_map = eval(eval(node_map['coords']))
            node_map['coords'] = ru.make_coordinates(coords_map)

    @staticmethod
    def check_type(atom): ## at list
        """
        """
        if atom['type'] == 'link':
            return 0
        elif atom['type'] == 'geom':
            return 1
        return -1

    def search_child_geometries(self, current):
        """
        """
        result = []
        for s in self.successors(current):
            if self.check_node_type(s) == 0: ## link
                pass
            else:
                result.append( [s] + self.search_child_geometries(s) )
        return result

    def search_parent_link(self, current):
        """
        """
        while True:
            pred = list(self.predecessors(current))
            if len(pred) > 0:
                current = pred[0]
                if self.check_node_type(current) == 0:
                    return current
            else:
                return None

    def search_direct_child_links(self, current):
        """
        """
        result = []
        for s in self.successors(current):
            if self.check_node_type(s) == 0: ## link
                result.append(s)
            else:
                res = self.search_direct_child_links(s)
                if len(res) > 0:
                    result.append(res)
        return result

    def check_node_type(self, node): ## at graph
        """
        """
        params=self.nodes[node]['params']
        return RobotTree.check_type(params)

    def down_convert_coords(self, scale=1.0):
        """
        Converting instance of coordinates class to dict (translation and rotation)
        """
        for nd in self.nodes:
            node_map = self.nodes[nd]
            if 'coords' in node_map:
                cds = node_map['coords']
                del node_map['coords']
                params = node_map['params']
                params.update(ru.make_coords_map(cds, method='rotation'))

    def up_convert_coords(self, force=False):
        for n in self.nodes:
            RobotTree._up_to_coordinates(self.nodes[n], force=force)

    def update_coords(self, absolute=True, force=False):
        """
        Updating coordinates in tree (set absolute coords to all nodes)
        """
        self.up_convert_coords(force=force)
        root  = self.search_root()
        leafs = self.search_leafs()
        updated = set()
        for l in leafs: ## ancestors
            cur = l
            lst = []
            while True:
                lst.append(cur)
                pred = list(self.predecessors(cur))
                if len(pred) < 1:
                    break
                cur=pred[0]
            ### lst is list from leaf to root
            ##print(lst)
            ## update loop
            cur_coords = coordinates()
            prev_node=None
            for node in reversed(lst):
                if absolute:
                    if node in updated:
                        cur_coords = self.nodes[node]['coords']
                    else:
                        updated.add(node)
                        p  = self.nodes[node]
                        pp = self.nodes[node]['params']
                        if 'relative' in pp and pp['relative']:
                            rel = p['coords'] ## relative
                            cur_coords = cur_coords.get_transformed(rel)
                            p['coords'] = cur_coords
                            del pp['relative']
                        else:
                            cur_coords = p['coords']
                else: ## relative
                    if node in updated:
                        rel = self.nodes[node]['coords']
                        cur_coords = cur_coords.get_transformed(rel)
                    else:
                        updated.add(node)
                        p  = self.nodes[node]
                        pp = self.nodes[node]['params']
                        if 'relative' in pp and pp['relative']:
                            rel = p['coords'] ## relative
                            cur_coords = cur_coords.get_transformed(rel)
                        else:
                            cds = p['coords']
                            p['coords'] = cur_coords.transformation(cds)
                            pp['relative'] = True
                            cur_coords = cds
                prev_node=node

    def _move_geom_parallel(self, link_node): ## internal use only
        geoms = self.search_child_geometries(link_node)
        for g in _flatten(geoms):
            self.remove_edges_to_node(g, fix=True)
            self.add_edge(link_node, g)

    def _move_geom_serial(self, link_node): ## internal use only
        geoms = self.search_child_geometries(link_node)
        cur_parent=link_node
        for g in _flatten(geoms):
            self.remove_edges_to_node(g, fix=True)
            self.add_edge(cur_parent, g)
            cur_parent=g

    def move_geometries_as_direct_children(self, method='parallel'): ## parallel or serial
        """
        """
        links = [ n for n in self.nodes if self.check_node_type(n) == 0 ]
        if method == 'parallel':
            method = self._move_geom_parallel
        elif method == 'serial':
            method = self._move_geom_serial
        for l in links:
            method(l)

    def add_robot_node(self, name, node_type, coords, **args):
        if self.has_node(name):
            return False
        ## not implemented yet
        return True

    def add_robot_edge(self, parent, child):
        ## not implemented yet
        if self.has_edge(parent, child):
            return False
        ## not implemented yet
        return True

    def insert_robot_node(self, parent, child, name, node_type, coords, **args):
        if self.has_node(name):
            return False
        ## not implemented yet
        return True
    #
    # DOT
    #
    def add_dot_shapes(self):
        """
        """
        for nd in self.nodes:
            node_attr = self.nodes[nd]
            params = node_attr['params']
            if params['type'] == 'geom':
                node_attr['shape'] = 'box'
            elif 'joint' in params and params['type'] == 'link' and params['joint']['type'] == 'root':
                node_attr['shape'] = 'diamond'
    #
    # Body
    # TODO: copy original geometry
    #
    def _parse_body(self, in_body):
        rt = in_body.rootLink
        for l in in_body.links:
            if l is rt:
                params={}
                params['node'] = l.name
                params['type'] = 'link'
                params['joint'] = {'type': 'root'}
                self.add_node(l.name, params=params, coords=coordinates(l.T))
            else:
                jtype = l.jointTypeString
                if jtype == 'revolute':
                    pass
                elif jtype == 'linear':
                    pass
                elif jtype == 'fixed':
                    pass
                else:
                    raise Exception('unknown joint type: {}'.format(jtype))
                params={}
                params['node'] = l.name
                params['type'] = 'link'
                jointd = {}
                jointd['type'] = jtype
                jointd['axis'] = l.jointAxis.tolist()
                jointd['id']   = l.jointId
                ## other parameters
                params['joint'] = jointd
                self.add_node(l.name, params=params, coords=coordinates(l.T))
                p = l.getParent()
                self.add_edge(p.name, l.name)
    @staticmethod
    def generate_from_body(in_body):
        """
        making graph(tree) from body

        Argas:
            in_bodyt ( cnoid.Body.Body ) : Input

        Retuns:
            RobotTree : Output

        """
        G = RobotTree()
        G._parse_body(in_body)
        return G

    #
    # geometry generation
    # ### generator(*1*)
    def _make_prev_joint_node(self, jnode, scale=1.0):
        node_map = self.nodes[jnode]
        params = node_map['params']
        tp = params['joint']['type']
        result = {'node': '{}_prev'.format(jnode), 'type': 'geom'}
        args = [ node_map['coords'] ]
        if 'axis' in params['joint']:
            args.append(params['joint']['axis'])
        (coords, geometry) = self.generator.generate('prev_'+ tp, *args, scale=scale)
        if coords is None:
            return (None, None)
        else:
            result['geometry'] = geometry
            #result['coords']   = coords
        return (result, coords)

    def _make_post_joint_node(self, jnode, scale=1.0):
        node_map = self.nodes[jnode]
        params = node_map['params']
        tp = params['joint']['type']
        result = {'node': '{}_post'.format(jnode), 'type': 'geom'}
        args = [ node_map['coords'] ]
        if 'axis' in params['joint']:
            args.append(params['joint']['axis'])
        (coords, geometry) = self.generator.generate('post_'+ tp, *args, scale=scale)
        if coords is None:
            return (None, None)
        else:
            result['geometry'] = geometry
            #result['coords']   = coords
        return (result, coords)

    def add_geometries_for_joints(self, scale=1.0):
        """
        """
        nodes = list(self.nodes)
        for node in nodes:
            if self.check_node_type(node) == 0: ## link
                succ = list(self.successors(node))
                pred = list(self.predecessors(node))
                if len(pred) > 0: ## ?
                    # add joint-geometry-prev
                    (j_pre_p, coords) = self._make_prev_joint_node(node, scale=scale)
                    if j_pre_p is not None:
                        j_pre = j_pre_p['node']
                        self.add_node(j_pre, params=j_pre_p, coords=coords)
                        self.remove_edge(pred[0], node)
                        self.add_edge(pred[0], j_pre)
                        self.add_edge(j_pre, node)
                else:
                    pass
                    ## warning not-tree
                # add joint-geometry-post
                (j_post_p, coords) = self._make_post_joint_node(node, scale=scale)
                if j_post_p is not None:
                    j_post = j_post_p['node']
                    self.add_node(j_post, params=j_post_p, coords=coords)
                    if len(succ) > 0:
                        self.remove_edge(node, succ[0])
                        self.add_edge(node, j_post)
                        self.add_edge(j_post, succ[0])
                    else:
                        self.add_edge(node, j_post)

    def _add_geometry_bw_links(self, parent, child, scale=1.0):
        p_params = self.nodes[parent]['params']
        c_params = self.nodes[child]['params']
        p_cds = self.nodes[parent]['coords']
        c_cds = self.nodes[child]['coords']
        (coords, geometry) = self.generator.generate("between_links", p_cds, c_cds, scale=scale)
        if coords is not None:
            name = 'geom_{}_{}'.format(parent, child)
            gparams = {'node': name, 'type': 'geom'}
            gparams['geometry'] = geometry
            #gparams['coords']   = coords
            self.add_node(name, params=gparams, coords=coords)
            self.add_edge(parent, name) ## parallel

    def add_geometries_for_links(self, scale=1.0):
        """
        """
        links = [ n for n in self.nodes if self.check_node_type(n) == 0 ]
        for l in links:
            clinks = _flatten(self.search_direct_child_links(l))
            for c in clinks:
                self._add_geometry_bw_links(l, c, scale=scale)

def makeShape(module, primitive=None, args=None, coords=None):
    """
    Args:
        module () :
        primitive (str) : Type of primitive
        args (dict) :
        coords (coordinates) :

    Returns:
    """
    func = None
    if primitive == 'box':
        func = module.makeBox
    elif primitive == 'cylinder':
        func = module.makeCylinder
    elif primitive == 'sphere':
        func = module.makeCylinder
    elif primitive == 'cone':
        func = module.makeCone
    elif primitive == 'capsule':
        func = module.makeCapsule
    elif primitive == 'torus':
        func = module.makeTorus
    elif primitive == 'tetrahedron':
        func = module.makeTetrahedron
    elif primitive == 'extrusion':
        func = module.makeExtrusion
    elif primitive == 'scen':
        ## not implemented yet
        pass
    elif primitive == 'mesh':
        ## not implemented yet
        pass
    elif primitive == 'scene_graph':
        if args is not None:
            pass ## add args['SgNode'] = xxx
    else:
        return
    ###
    ret = func(**args)
    if coords is not None:
        ret.newcoords(coords)
    return ret

class RobotTreeBuilder(RobotBuilder):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_density = 1000
        self.default_joint_range    = [-PI, PI]
        self.default_velocity_range = [-PI*10, PI*10]
        self.default_effort_range   = [-100, 100]
        self.scale = 1.0
    #
    def makeShapeFromParam(self, primitive=None, args=None, coords=None):
        return makeShape(self, primitive=primitive, args=args, coords=coords)

#    def makeShapeFromParam(self, primitive=None, args=None, coords=None):
#        func = None
#        if primitive == 'box':
#            func = self.makeBox
#        elif primitive == 'cylinder':
#            func = self.makeCylinder
#        elif primitive == 'sphere':
#            func = self.makeCylinder
#        elif primitive == 'cone':
#            func = self.makeCone
#        elif primitive == 'capsule':
#            func = self.makeCapsule
#        elif primitive == 'torus':
#            func = self.makeTorus
#        elif primitive == 'tetrahedron':
#            func = self.makeTetrahedron
#        elif primitive == 'extrusion':
#            func = self.makeExtrusion
#        elif primitive == 'scene_graph':
#            if args is not None:
#                pass ## add args['SgNode'] = xxx
#        else:
#            return
#        ###
#        ret = func(**args)
#        if coords is not None:
#            ret.newcoords(coords)
#        return ret

    def _buildGeomForLink(self, rbt, link_node):
        geoms = rbt.search_child_geometries(link_node)
        for g in _flatten(geoms):
            node_map = rbt.nodes[g]
            params = node_map['params']
            self.makeShapeFromParam(**params['geometry'], coords=node_map['coords'])

    def _addRoot(self, rbt, root_node):
        args = {}
        return self.createLinkFromShape(name=root_node,
                                        root=True,
                                        **args)

    def _addJoint(self, rbt, parent_lk, link_node):
        ## build geometry
        self._buildGeomForLink(rbt, link_node)

        ## build joint
        node_map = rbt.nodes[link_node]
        cds = node_map['coords']
        params = node_map['params']
        jp = params['joint']
        jtype = jp['type']

        if jtype == 'revolute' or jtype == 'revolute_yaw':
            j  = self.createJointShape(jointType=Link.JointType.RevoluteJoint)
        elif jtype == 'linear':
            j  = self.createJointShape(Link.JointType.PrismaticJoint)
        elif jtype == 'fixed':
            j  = self.createJointShape(Link.JointType.FixedJoint)
        elif jtype == 'ball':
            j  = self.createJointShape(RobotBuilder.JointType.Ball)
        ##
        j.newcoords(cds)
        if 'axis' in jp:
            axis = jp['axis']
            trs = _axis_to_rotation(axis)
            j.transform(trs)
        ##
        name  =link_node
        jname = link_node
        if 'name' in jp:
            jname = jp['name']
        args = {}
        if 'args' in jp:
            args = jp['args']
        lk = self.createLinkFromShape(name=name,
                                      JointName=jname,
                                      parentLink = parent_lk,
                                      JointId = self.cur_id,
                                      density = self.default_density,
                                      **args)
        if jtype == 'fixed':
            pass
        elif jtype == 'ball':
            self.cur_id += 3
        else:
            self.cur_id += 1

        ## build children
        link_nodes = _flatten(rbt.search_direct_child_links(link_node))
        for l in link_nodes:
            self._addJoint(rbt, lk, l)

    def buildRobotFromTree(self, robot_tree):
        """
        Args:
            robot_tree (RobotTree) : Input tree

        """
        self.cur_id = 0
        root_node = robot_tree.search_root()
        ## build root
        self._buildGeomForLink(robot_tree, root_node)
        root_lk = self._addRoot(robot_tree, root_node)
        ## build children
        link_nodes = _flatten(robot_tree.search_direct_child_links(root_node))
        for node in link_nodes:
            self._addJoint(robot_tree, root_lk, node)