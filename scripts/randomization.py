
import os
import numpy
import copy
import bpy
import bpy_extras

from math      import radians, sin, cos, tan
from mathutils import Vector, Matrix, Euler

# TODO: Configure the camera settings in the script
# TODO: Adjust code to support Cycles
# TODO: Function to change color with some noise (create different shades for the obj's original base color)

def create_new_camera():
    
    camera_data = bpy.data.cameras.new('Camera')
    camera      = bpy.data.objects.new('Camera', camera_data)

    return camera


def unlink_scene_objects(assets, keep_list=[]):
    for asset_dict in assets:
        for obj in asset_dict:
            if obj.name not in keep_list: 
                if asset_dict is bpy.data.materials:
                    if obj.texture_slots:
                        for i in range(len(obj.texture_slots)):
                            obj.texture_slots.clear(i)
                asset_dict.remove(obj, do_unlink=True)
            
    return True


def start_new_scene(resolution, use_cycles=False):  
    scene  = bpy.data.scenes[0]    
    camera = create_new_camera()
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    
    if use_cycles:
        scene.view_layers[0].cycles.use_denoising = True
    
    world = bpy.data.worlds.new('World')
    #world.use_sky_paper = True
    scene.world = world
    scene.view_layers[0].update()
    
    return scene, camera, world


def import_models_assets(cad_path, main_obj_name, contextual_distractors=[]):
    
    main_obj = []
    ctx_distractors = []
    
    for obj_name in ([main_obj_name] + contextual_distractors):
        obj_path = os.path.join(cad_path, f'{obj_name}.obj')
        bpy.ops.import_scene.obj(filepath=obj_path)
        obj = bpy.context.selected_objects[0]
        obj.name = obj_name
        obj.data.name = obj_name
        if obj_name == main_obj_name:
            main_obj = obj
            if obj_name not in bpy.data.scenes[0].objects.keys():
                bpy.data.scenes[0].collection.objects.link(bpy.data.objects[obj_name])
        else:
            ctx_distractors.append(obj)
        
        if obj.data.materials:
            obj.data.materials[0].name = obj_name
        
    return main_obj, ctx_distractors
            

def apply_texture(obj, texture_file, name='Texture'):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes['Principled BSDF']
    texImage = material.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(texture_file)
    material.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    
    return material


def apply_solid_color(obj, name='Material'):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes['Principled BSDF']
    mat_output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(bsdf.inputs['Base Color'], mat_output.inputs['Surface'])
    
    R = numpy.random.uniform(low=0, high=1)
    G = numpy.random.uniform(low=0, high=1)
    B = numpy.random.uniform(low=0, high=1)
    
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    
    obj.data.materials[0].node_tree.nodes['Principled BSDF'].inputs[0].default_value = (R, G, B, 1)
    
    return material


def change_material_base_color(obj, RGB=None):
    
    if RGB:
        RGBA = (RGB[0], RGB[1], RGB[2], 1)
    
    else:
        R    = numpy.random.uniform(low=0, high=1)
        G    = numpy.random.uniform(low=0, high=1)
        B    = numpy.random.uniform(low=0, high=1)
        RGBA = (R, G, B, 1)
    
    obj.data.materials[0].node_tree.nodes['Principled BSDF'].inputs[0].default_value = RGBA

    return True


def create_ground_plane(ground_plane_size, ground_textures=None):
    bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
    bpy.ops.transform.resize(value=(ground_plane_size, ground_plane_size, 0))
    ground = bpy.context.active_object
    
    if ground_textures:
        image_index = numpy.random.randint(low=1, high=3361)
        tex_path    = os.path.join(ground_textures, f'texture ({image_index}).jpg')
        apply_texture(ground, tex_path, name='Ground')

    return ground


def create_skybox(skybox_plane_length, skybox_textures_path, apply_same_texture=True, include_top_plane=True):
    
    box = {
    #'TOP':    [(0, 0,  skybox_plane_length), (0.,  0., 0.)],
    #'BOTTOM': [(0, 0, -skybox_plane_length), (0.,  0., 0.)],
    'FRONT':  [( skybox_plane_length, 0, 0), (0., 90., 0.)],
    'BACK':   [(-skybox_plane_length, 0, 0), (0., 90., 0.)],
    'LEFT':   [(0,  skybox_plane_length, 0), (90., 0., 0.)],
    'RIGTH':  [(0, -skybox_plane_length, 0), (90., 0., 0.)]
    }
    
    if include_top_plane:
        box['TOP'] = [(0, 0,  skybox_plane_length), (0.,  0., 0.)]
    
    material = None
    skybox   = list()
    
    for idx, side in enumerate(box):
        bpy.ops.mesh.primitive_plane_add(location=box[side][0])
        bpy.ops.transform.resize(value=(skybox_plane_length, skybox_plane_length, 0))
        plane = bpy.context.active_object
        plane.rotation_euler = ([radians(a) for a in box[side][1]])
        
        if apply_same_texture and material != None:
            if plane.data.materials:
                plane.data.materials[0] = material
            else:
                plane.data.materials.append(material)
        else:
            idx = numpy.random.randint(low=1, high=3361)
            tex = os.path.join(skybox_textures_path, f'texture ({idx}).jpg')
            material = apply_texture(plane, tex, name=side)
        skybox.append(plane)
    
    return skybox

def set_object_position(obj, type='center', resize_factor=(1.,1.,1.), limits=[0.,0.,0.]):
    
    if type == 'center':
        obj.location = (0., 0., 0.)
        
    elif type == 'random':
        x = numpy.random.uniform(low=-limits[0], high=limits[0] + 1)
        y = numpy.random.uniform(low=-limits[1], high=limits[1] + 1)
        z = numpy.random.uniform(low=0., high=limits[2]/2 + 1)
        obj.location = (x, y, z)
    
    else:
        print('Unexpected type for the object location')
        return False
    
    bpy.ops.transform.resize(value=resize_factor)
    obj.rotation_euler = ([radians(a) for a in (0, 0, 0)])
    
    return True


def generate_flying_distractors(objects_list, texture_path = None, num_objs=[0, 2], limits=[0.,0.,0.], scale_range=[1.,2.], rotation_limits=[0,360], material_type='solid', projections=None):
    
    distractors = list()
    
    N = numpy.random.randint(low=num_objs[0], high=num_objs[1] + 1)
    
    for i in range(N):
        
        x = numpy.random.uniform(low=-limits[0], high=limits[0] + 1)
        y = numpy.random.uniform(low=-limits[1], high=limits[1] + 1)
        z = numpy.random.uniform(low=0., high=limits[2]/2 + 1)
        location = (x, y, z)
        
        sx = numpy.random.uniform(low=scale_range[0], high=scale_range[1] + 1)
        sy = numpy.random.uniform(low=scale_range[0], high=scale_range[1] + 1)
        sz = numpy.random.uniform(low=scale_range[0], high=scale_range[1] + 1)
        
        rx = numpy.random.randint(low=rotation_limits[0], high=rotation_limits[1])
        ry = numpy.random.randint(low=rotation_limits[0], high=rotation_limits[1])
        rz = numpy.random.randint(low=rotation_limits[0], high=rotation_limits[1])
        rotation = ([radians(a) for a in (rx, ry, rz)])
        
        object_index = numpy.random.randint(low=0, high=len(objects_list))
        object_type = objects_list[object_index]   
        
        if object_type == 'CUBE':    
            bpy.ops.mesh.primitive_cube_add(location=location)
            
        elif object_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(location=location)
            
        elif object_type == 'CONE':
            bpy.ops.mesh.primitive_cone_add(location=location)
            
        elif object_type == 'TORUS':
            bpy.ops.mesh.primitive_torus_add(location=location)
            
        elif object_type == 'CYLINDER':
            bpy.ops.mesh.primitive_cylinder_add(location=location)
                        
        elif object_type == 'MONKEY':
            bpy.ops.mesh.primitive_monkey_add(location=location)
        
        distractor = bpy.context.active_object
        bpy.ops.transform.resize(value=(sx, sy, sz))
        distractor.rotation_euler = rotation
        distractor.name = f'Distractor {i}'
        distractors.append(distractor)
        
        mat_type = material_type
        
        if texture_path is None and (material_type == 'texture' or material_type == 'both'):
            mat_type = 'solid'
        
        if mat_type == 'both':
            prob = numpy.random.uniform()
            mat_type = 'solid' if prob < 0.5 else 'texture'
            
        if mat_type == 'texture':
            image_index = numpy.random.randint(low=1, high=3361)
            texture_file    = os.path.join(texture_path, f'texture ({image_index}).jpg')
            apply_texture(distractor, texture_file, name=f'Distractor {i}')
        
        elif mat_type == 'solid':
            apply_solid_color(distractor, name=f'Distractor {i}')
        
    return distractors
        

def get_random_camera_motion(steps=1, ref_obj=None, min_distance=1, max_distance=2, apply_tilt=False, tilt=[-1,1], pan=[-1,1], roll=[-1,1]):
    
    locations = list()
    rotations = list()
    
    for _ in range(steps):
        radius      = numpy.random.randint(low=min_distance, high=max_distance + 1)
        inclination = radians(numpy.random.randint(low=1, high=90))
        azimuth     = radians(numpy.random.randint(low=0, high=360))
        
        x = radius * sin(inclination) * cos(azimuth)
        y = radius * sin(inclination) * sin(azimuth)
        z = radius * cos(inclination)

        location = Vector([x, y, z])

        rotation_base = Vector([0,0,0])
        
        # Look at the reference object's direction
        if ref_obj:
            obj_loc       = ref_obj.matrix_world.to_translation()
            direction     = obj_loc - location
            rotation_q    = direction.to_track_quat('-Z', 'Y')            
            rotation_base = rotation_q.to_euler()    

        if apply_tilt:
            tilt_ = numpy.random.randint(low=tilt[0], high=tilt[1] + 1)
            pan_  = numpy.random.randint(low=pan[0], high=pan[1] + 1)
            roll_ = numpy.random.randint(low=roll[0], high=roll[1] + 1)

            rotation_shift = Vector([radians(a) for a in (tilt_, pan_, roll_)])
            rotation_base  = Euler(Vector(rotation_base) + rotation_shift)        

        locations.append(location)
        rotations.append(rotation_base)
        
    return locations, rotations


def get_random_camera_motion_in_dist_range(steps=1, ref_obj=None, min_distance=1, max_distance=10, distance_step=1, keep_view=True, apply_tilt=False, tilt=[-1,1], pan=[-1,1], roll=[-1,1]):
    ''' It is similar to the function above, do random movements in a spherical path
    but keeping a range of different distances in the same position'''
    
    assert min_distance < max_distance, 'minimum distance cannot be greater than maximum distance'
    
    locations = list()
    rotations = list()
    
    for _ in range(steps):
        
        inclination = radians(numpy.random.randint(low=1, high=90))
        azimuth     = radians(numpy.random.randint(low=0, high=360))
        
        for radius in range(min_distance, max_distance, distance_step):
            
            if not keep_view:
                inclination = radians(numpy.random.randint(low=1, high=90))
                azimuth     = radians(numpy.random.randint(low=0, high=360))
                
            x = radius * sin(inclination) * cos(azimuth)
            y = radius * sin(inclination) * sin(azimuth)
            z = radius * cos(inclination)

            location = Vector([x, y, z])

            rotation_base = Vector([0,0,0])
            
            # Look at the reference object's direction
            if ref_obj:
                obj_loc       = ref_obj.matrix_world.to_translation()
                direction     = obj_loc - location
                rotation_q    = direction.to_track_quat('-Z', 'Y')            
                rotation_base = rotation_q.to_euler()    

            if apply_tilt:
                tilt_ = numpy.random.randint(low=tilt[0], high=tilt[1] + 1)
                pan_  = numpy.random.randint(low=pan[0], high=pan[1] + 1)
                roll_ = numpy.random.randint(low=roll[0], high=roll[1] + 1)

                rotation_shift = Vector([radians(a) for a in (tilt_, pan_, roll_)])
                rotation_base  = Euler(Vector(rotation_base) + rotation_shift)        

            locations.append(location)
            rotations.append(rotation_base)
            
    return locations, rotations
        

def get_circular_camera_motion(steps=1, ref_obj=None, min_distance=1, max_distance=2, apply_tilt=False, tilt=[-1,1], pan=[-1,1], roll=[-1,1]):
    pass            


def create_light(name, type, **light_prop):
    light_data = bpy.data.lights.new(name=f'{name}_data', type=type)
    light_obj  = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    
    if 'energy' in light_prop.keys():
        light_data.energy = light_prop['energy']
    
    if 'location' in light_prop.keys():
        light_obj.location = light_prop['location']
    
    if 'rotation' in light_prop.keys():
        light_obj.rotation_euler = light_prop['rotation']
    
    if 'color' in light_prop.keys():
        light_data.color = light_prop['color']
    
    if 'radius' in light_prop.keys():
        light_data.shadow_soft_size = light_prop['radius']
    
    if 'angle' in light_prop.keys():
        light_data.angle = radians(light_prop['angle'])
    
    return light_obj


def create_sun(energy_range, angle_range, height, pos_range=[0.,0.], change_sun_pos=True, change_sun_rot=True):

    sun_energy   = numpy.random.uniform(low=energy_range[0], high=energy_range[1])
    sun_angle    = numpy.random.uniform(low=angle_range[0],  high=angle_range[1])
    
    if change_sun_pos:
        sunx         = numpy.random.uniform(low=-pos_range[0], high=pos_range[0])
        suny         = numpy.random.uniform(low=-pos_range[1], high=pos_range[1])
        sun_location = (sunx, suny, height)
    else:
        sun_location = (0., 0., height)
    
    if change_sun_rot:    
        sunrx        = numpy.random.uniform(low=0, high=90)
        sunry        = numpy.random.uniform(low=0, high=90)
        sunrz        = numpy.random.uniform(low=0, high=90)
        sun_rotation = tuple([radians(a) for a in (sunrx, sunry, sunrz)])
    else:
        sun_rotation = (0., 0., 0.)

    sun = create_light('SUN', 'SUN', energy=sun_energy, location=sun_location, rotation=sun_rotation, angle=sun_angle)

    return sun


def create_environment_lights(num_lights_range, energy_range, radius_range, pos_range, height, change_color=True):
    
    n_lights = numpy.random.randint(low=num_lights_range[0], high=num_lights_range[1])
    
    lights = list()
    
    for i in range(n_lights):
        
        energy   = numpy.random.uniform(low=energy_range[0], high=energy_range[1])
        radius   = numpy.random.uniform(low=radius_range[0], high=radius_range[1])
        x        = numpy.random.uniform(low=-pos_range[0], high=pos_range[0])
        y        = numpy.random.uniform(low=-pos_range[1], high=pos_range[1])
        z        = numpy.random.uniform(low=1, high=height)
        location = (x, y, z)
        
        if change_color:
            R = numpy.random.uniform(low=0, high=1)
            G = numpy.random.uniform(low=0, high=1)
            B = numpy.random.uniform(low=0, high=1)
            color = (R, G, B)
        else:
            color = (1., 1., 1.)
        
        light = create_light(f'LIGHT{i}', 'POINT', energy=energy, radius=radius, color=color, location=location)
        lights.append(light)
    
    return lights
    

def get_2d_coordinates(scene, cam, obj, get_only_corners=True):
    
    vertices = (vert.co for vert in obj.data.vertices)
        
    obj_matrix = obj.matrix_world
    
    if get_only_corners:
        minx, miny, minz, maxx, maxy, maxz = 1000, 1000, 1000, -1, -1, -1
        for coord in vertices:
            minx = coord[0] if coord[0] < minx else minx
            miny = coord[1] if coord[1] < miny else miny
            minz = coord[2] if coord[2] < minz else minz
            maxx = coord[0] if coord[0] > maxx else maxx
            maxy = coord[1] if coord[1] > maxy else maxy
            maxz = coord[2] if coord[2] > maxz else maxz
        
        vertices = [Vector([minx, miny, minz]), Vector([minx, miny, maxz]),
                    Vector([minx, maxy, minz]), Vector([minx, maxy, maxz]),
                    Vector([maxx, miny, minz]), Vector([maxx, miny, maxz]),
                    Vector([maxx, maxy, minz]), Vector([maxx, maxy, maxz])]
            
    coords = [bpy_extras.object_utils.world_to_camera_view(scene, cam, obj_matrix @ coord) for coord in vertices]
    scale  = scene.render.resolution_percentage / 100.
    
    rx = scene.render.resolution_x * scale
    ry = scene.render.resolution_y * scale
    
    points2D = list()
    for x, y, d in coords:
        if d > 0: # get the points that are in front of the camera
            points2D.append((x*rx, ry - y*ry))
    
    return points2D


def write_keyframe(filename, scene):
    scene.render.filepath = filename
    scene.render.resolution_percentage = 100
    bpy.ops.render.render(write_still=True)
    return True


def main():
    
    # Engine parameters
    TOTAL_NUM_FRAMES  = 20 
    OCCLUSION_PERCENT = 1.
    USE_CYCLES        = False # Actually, it only supports eevee textures (blender default)
    SAMPLES_PER_FRAME = 64    # Applied only if using cycles 
    
    # Paths
    CAD_PATH          = os.path.join(os.path.dirname(bpy.data.filepath), 'meshes')
    MAIN_OBJ_NAME     = 'cat_red'
    DISTRACTORS_NAMES = ''
    KEEP_LIST         = [MAIN_OBJ_NAME] 
    SKYBOX_TEXTURES   = os.path.join(os.path.dirname(bpy.data.filepath), 'textures', 'mscoco')
    GROUND_TEXTURES   = os.path.join(os.path.dirname(bpy.data.filepath), 'textures', 'dtd')
    #OUTPUT_IMAGES     = os.path.join(os.path.dirname(bpy.data.filepath), 'render', 'output %d.jpg')
    OUTPUT_IMAGES     = os.path.join(os.path.dirname(bpy.data.filepath), 'render/images', 'output %d.jpg')
    OUTPUT_LABELS     = OUTPUT_IMAGES.replace('jpg', 'txt').replace('images', 'corners')
    SAVE_ONLY_CORNERS = True     # If True, it saves only the projected points for the 3d bounding box. If false, it saves all projected model vertices
    
    #environment parameters
    ENVIRONMENT        = 'indoor' # choose indoor or outdoor to create or not the skybox (impacts the sunlight if it is out of skybox range)
    SKYBOX_LENGTH      = 50
    GROUND_PLANE_LEN   = 50
    APPLY_SAME_TEX_SKY = True
    INCLUDE_TOP_SKYBOX = True
    
    SUN_MIN_ANGLE     = 2
    SUN_MAX_ANGLE     = 10
    SUN_MIN_ENERGY    = 1.
    SUN_MAX_ENERGY    = 3.
    CHANGE_SUN_POS    = True
    SUN_MIN_POS       = -SKYBOX_LENGTH
    SUN_MAX_POS       = SKYBOX_LENGTH
    SUN_HEIGHT        = SKYBOX_LENGTH - 2
    CHANGE_SUN_ROT    = True
    
    MAX_NUM_LIGHTS     = 20
    MIN_NUM_LIGHTS     = 5
    LIGHT_MIN_RADIUS   = 0.2
    LIGHT_MAX_RADIUS   = 1.5
    LIGHT_MIN_ENERGY   = 50
    LIGHT_MAX_ENERGY   = 250
    MIN_STRENGTH       = 10
    MAX_STRENGTH       = 250
    CHANGE_LIGHT_COLOR = True
    LIGHT_MIN_POS      = -SKYBOX_LENGTH / 3.
    LIGHT_MAX_POS      = SKYBOX_LENGTH / 3.
    LIGHT_MAX_HEIGHT   = SKYBOX_LENGTH / 2.
    
    # Camera Parameters
    RES = (680, 680)
    FOV = 30
    SENSOR_SIZE = 35
    ASPECT = 1.
    FX, FY = 1000, 1000
    U0, V0 = 500,  500
    
    CAMERA_MAX_DISTANCE  = SKYBOX_LENGTH - 2
    CAMERA_MIN_DISTANCE  = 20
    CAMERA_DISTANCE_STEP = 5 
    TILT_RANGE   = [-1,1]
    PAN_RANGE    = [-1,1]
    ROLL_RANGE   = [-1,1]
    APPLY_TILT   = True
    MOTION_TYPE  = 'simple-random'  # Choose between 'simple-random', 'radius-range', or 'circular-path' #TODO: circular-path need adjustments
    KEEP_VIEW    = True             # If using 'radius-range' it will keep the same inclination and azimuth for different radius. If False it will change the view and distance keeping the object setup (like simple-random but increasing the camera distance each step)
    MOTION_STEPS = 1                # Get different camera locations (views) for the same configuration of object, lights, and distractors
    
    # Mesh parameters
    INITIAL_LOCATION   = 'center' # use center or random
    RESIZE_MESH_FACTOR = (.25, .25, .25)
    POS_LIMITS         = [SKYBOX_LENGTH/2, SKYBOX_LENGTH/2, SKYBOX_LENGTH/2]
    CHANGE_OBJ_COLOR   = True
    
    # Distractors
    NUM_DIST            = [1, 10]
    DIST_POS_LIMITS     = POS_LIMITS
    DIST_ROTATIONS      = [0, 360]
    DIST_SCALE          = [1., 1.]
    DIST_PROJECTIONS    = None       # if using cycles
    DIST_LIST           = ['CUBE', 'SPHERE', 'CONE', 'TORUS', 'CYLINDER', 'MONKEY']
    DIST_TEX            = os.path.join(os.path.dirname(bpy.data.filepath), 'textures', 'dtd')
    DIST_MAT_TYPE       = 'both' # choose between 'solid', 'texture', or 'both'
    
    # Assets
    ASSETS = [bpy.data.objects, 
    bpy.data.lights,
    bpy.data.worlds,
    bpy.data.textures,
    bpy.data.images,
    bpy.data.materials,
    bpy.data.cameras,
    bpy.data.meshes]
    
    # Start scene and intialize objects
    unlink_scene_objects(ASSETS, [])
    
    saved_keyframes_counter = 0
    
    scene, camera, world = start_new_scene(RES, use_cycles=USE_CYCLES)
    main_obj, _  = import_models_assets(CAD_PATH, MAIN_OBJ_NAME)
    set_object_position(main_obj, INITIAL_LOCATION, RESIZE_MESH_FACTOR, POS_LIMITS)
    
    camera_name = camera.name
    KEEP_LIST = KEEP_LIST + [main_obj.data.materials[0].name, camera.name, world.name]
    
    # Do the randomization
    while saved_keyframes_counter < TOTAL_NUM_FRAMES:
        scene    = bpy.context.scene
        main_obj = bpy.data.objects[MAIN_OBJ_NAME]
        camera   = bpy.data.objects[camera_name]
        
        if ENVIRONMENT == 'indoor':
            skybox = create_skybox(SKYBOX_LENGTH, SKYBOX_TEXTURES, apply_same_texture=APPLY_SAME_TEX_SKY, include_top_plane=INCLUDE_TOP_SKYBOX)
        
        ground = create_ground_plane(GROUND_PLANE_LEN, GROUND_TEXTURES)
        
        distractors = generate_flying_distractors(DIST_LIST, texture_path=DIST_TEX, num_objs=NUM_DIST, limits=DIST_POS_LIMITS, 
                        scale_range=DIST_SCALE, rotation_limits=DIST_ROTATIONS, material_type=DIST_MAT_TYPE, projections=None)    
    
        sun = create_sun([SUN_MIN_ENERGY, SUN_MAX_ENERGY], [SUN_MIN_ANGLE, SUN_MAX_ANGLE], 
                    SUN_HEIGHT, [SUN_MIN_POS, SUN_MAX_POS], True, True)

        lights = create_environment_lights([MIN_NUM_LIGHTS, MAX_NUM_LIGHTS], [LIGHT_MIN_ENERGY, LIGHT_MAX_ENERGY], 
                            [LIGHT_MIN_RADIUS, LIGHT_MAX_RADIUS], [LIGHT_MIN_POS, LIGHT_MAX_POS], LIGHT_MAX_HEIGHT, change_color=CHANGE_LIGHT_COLOR)
        
        if MOTION_TYPE == 'simple-random':
            cam_locations, cam_rotations = get_random_camera_motion(steps=MOTION_STEPS, ref_obj=main_obj, 
                            min_distance=CAMERA_MIN_DISTANCE, max_distance=CAMERA_MAX_DISTANCE, 
                            apply_tilt=APPLY_TILT, tilt=TILT_RANGE, pan=PAN_RANGE, roll=ROLL_RANGE)
        
        elif MOTION_TYPE == 'radius-range':
            cam_locations, cam_rotations = get_random_camera_motion_in_dist_range(steps=MOTION_STEPS, ref_obj=main_obj, 
                            min_distance=CAMERA_MIN_DISTANCE, max_distance=CAMERA_MAX_DISTANCE, distance_step=CAMERA_DISTANCE_STEP, 
                            keep_view=KEEP_VIEW, apply_tilt=APPLY_TILT, tilt=TILT_RANGE, pan=PAN_RANGE, roll=ROLL_RANGE)

        if CHANGE_OBJ_COLOR:
            change_material_base_color(main_obj)

        # Apply the camera positions to render the randomized frames
        for c_loc, c_rot in zip(cam_locations, cam_rotations):
            camera.location       = c_loc
            camera.rotation_euler = c_rot
            
            # The object color can be changed here
            # If using multiple distances or steps, it'll create the same frame setup but with a different color for the main object
            #if CHANGE_OBJ_COLOR:
            #    change_material_base_color(main_obj)
            
            write_keyframe(OUTPUT_IMAGES % saved_keyframes_counter, scene)

            points = get_2d_coordinates(scene, camera, main_obj, get_only_corners=SAVE_ONLY_CORNERS)
            try:
                os.makedirs(os.path.dirname(OUTPUT_LABELS))
            except:
                pass
            numpy.savetxt(OUTPUT_LABELS % saved_keyframes_counter, numpy.array(points))
            
            OUT_VERTICES = OUTPUT_LABELS.replace('corners', 'projected_vertices') 
            try:
                os.makedirs(os.path.dirname(OUT_VERTICES))
            except:
                pass
            vert2d = get_2d_coordinates(scene, camera, main_obj, get_only_corners=False)
            numpy.savetxt( OUT_VERTICES % saved_keyframes_counter, numpy.array(vert2d))

            saved_keyframes_counter = saved_keyframes_counter + 1
            
            if saved_keyframes_counter >= TOTAL_NUM_FRAMES:
                break
        
        unlink_scene_objects(ASSETS, KEEP_LIST)

if __name__ == '__main__':
    main()