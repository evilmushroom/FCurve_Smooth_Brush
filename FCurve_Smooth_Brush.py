bl_info = {
    "name": "FCurve Smooth Brush",
    "author": "Evilmushroom/Claude",
    "version": (1, 6),
    "blender": (3, 6, 0),
    "location": "Graph Editor > Sidebar > Tool",
    "description": "Paint to smooth FCurves in the Graph Editor",
    "doc_url": "https://github.com/evilmushroom/FCurve_Smooth_Brush",
    "category": "Animation",
}

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy.props import FloatProperty, IntProperty, BoolProperty, EnumProperty
from mathutils import Vector
import blf
import time

def draw_brush_cursor(self, context, event):
    props = context.scene.fcurve_smooth_brush
    radius = props.brush_size
    
    # Single cursor circle
    segments = 32
    center = (self.last_mouse_region_x, self.last_mouse_region_y)
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    
    # Create circle vertices
    vertices = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    
    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.8))
    batch.draw(shader)
    
    # Draw center dot
    dot_vertices = []
    dot_radius = 2
    for i in range(16):
        angle = 2 * np.pi * i / 16
        x = center[0] + dot_radius * np.cos(angle)
        y = center[1] + dot_radius * np.sin(angle)
        dot_vertices.append((x, y))
    
    dot_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": dot_vertices})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
    dot_batch.draw(shader)

def draw_brush_callback_px(self, context):
    props = context.scene.fcurve_smooth_brush
    radius = props.brush_size
    strength = props.strength
    segments = 32
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    
    # Draw brush circle with gradient
    center = (self.mouse_pos.x, self.mouse_pos.y)
    vertices = []
    indices = []
    
    vertices.append(center)
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
        
        if i < segments - 1:
            indices.extend([0, i + 1, i + 2])
        else:
            indices.extend([0, segments, 1])
    
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    
    # Simple gradient
    for i in range(4):
        alpha = 0.15 * (1 - i/4) * strength
        shader.uniform_float("color", (1.0, 1.0, 1.0, alpha))
        batch.draw(shader)

# Then define your classes
class FCurveSmoothBrushProperties(bpy.types.PropertyGroup):
    brush_size: FloatProperty(
        name="Brush Size",
        description="Size of the brush",
        default=50.0,
        min=1.0,
        max=500.0,
        subtype='FACTOR'  # Makes it display as a slider
    )
    strength: FloatProperty(
        name="Strength",
        description="Strength of the brush effect",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR'  # Makes it display as a slider
    )
    iterations: IntProperty(
        name="Iterations",
        description="Number of iterations",
        default=1,
        min=1,
        max=10
    )
    brush_mode: EnumProperty(
        name="Brush Mode",
        description="Brush operation mode",
        items=[
            ('SMOOTH', "Smooth", "Smooth keyframes"),
            ('NOISE', "Noise", "Add controlled noise"),
            ('FLATTEN', "Flatten", "Flatten to average value"),
            ('SHARPEN', "Sharpen", "Increase contrast between keyframes"),
            ('RELAX', "Relax", "Evenly space keyframes")
        ],
        default='SMOOTH'
    )
    affect_selected: BoolProperty(
        name="Selected Only",
        description="Only affect selected keyframes",
        default=False
    )
    select_while_painting: BoolProperty(
        name="Select While Painting",
        description="Auto-select keyframes under brush",
        default=False
    )
    use_acceleration: BoolProperty(
        name="Use Acceleration",
        description="Speed up brush for dense keyframes",
        default=True
    )
    sample_rate: IntProperty(
        name="Sample Rate",
        description="Process every Nth keyframe",
        default=1,
        min=1,
        max=10
    )
    auto_frame: BoolProperty(
        name="Auto Frame",
        description="Auto frame affected area",
        default=False
    )
    preserve_handles: BoolProperty(
        name="Preserve Handles",
        description="Maintain handle types",
        default=True
    )
    is_active: BoolProperty(
        name="Brush Active",
        description="Whether brush is active",
        default=False
    )

class FCurveSmoothBrushOperator(bpy.types.Operator):
    bl_idname = "graph.fcurve_smooth_brush"
    bl_label = "FCurve Smooth Brush"
    bl_options = {'REGISTER', 'UNDO'}
    
    def __init__(self):
        self.mouse_pos = Vector((0, 0))
        self._handle = None
        self._cursor_handle = None
        self.is_painting = False
        self.stroke_buffer = []
        self.last_mouse_region_x = 0
        self.last_mouse_region_y = 0
        self.last_process_time = 0
        self.process_interval = 0.032
        self.keyframe_cache = {}
        self.stroke_start_values = {}
        self.active_stroke = False
        self.undo_states = []
        self.max_undo_states = 32  # Limit memory usage
        
    def clear_cache(self):
        """Clear all cached data to prevent memory leaks"""
        self.keyframe_cache.clear()
        self.stroke_buffer.clear()
        
    def cleanup_handlers(self):
        """Remove draw handlers safely"""
        if self._handle is not None:
            try:
                bpy.types.SpaceGraphEditor.draw_handler_remove(self._handle, 'WINDOW')
            except:
                pass
            self._handle = None
            
        if self._cursor_handle is not None:
            try:
                bpy.types.SpaceGraphEditor.draw_handler_remove(self._cursor_handle, 'WINDOW')
            except:
                pass
            self._cursor_handle = None
        self.undo_states.clear()
        self.stroke_buffer.clear()
        self.stroke_start_values.clear()

    def store_undo_state(self):
        """Store current state of all relevant fcurves"""
        state = {}
        for fcurve in bpy.context.selected_editable_fcurves:
            if not fcurve.hide:
                # Store a deep copy of keyframe data
                state[fcurve] = [(kf.co[0], kf.co[1], 
                                kf.handle_left[:], kf.handle_right[:],
                                kf.handle_left_type, kf.handle_right_type)
                               for kf in fcurve.keyframe_points]
        
        self.undo_states.append(state)
        # Maintain fixed size undo buffer
        if len(self.undo_states) > self.max_undo_states:
            self.undo_states.pop(0)

    def restore_state(self, state):
        """Restore fcurves to given state"""
        for fcurve, keyframe_data in state.items():
            if fcurve.id_data:  # Check if fcurve is still valid
                for i, (x, y, hl, hr, hlt, hrt) in enumerate(keyframe_data):
                    if i < len(fcurve.keyframe_points):
                        kf = fcurve.keyframe_points[i]
                        kf.co = (x, y)
                        kf.handle_left = hl
                        kf.handle_right = hr
                        kf.handle_left_type = hlt
                        kf.handle_right_type = hrt
                fcurve.update()

    def begin_stroke(self):
        """Initialize stroke state"""
        self.active_stroke = True
        self.stroke_buffer.clear()
        self.store_undo_state()  # Store state before modifications
        
    def end_stroke(self, context):
        """Finalize stroke"""
        if self.active_stroke:
            self.active_stroke = False
            # Don't use Blender's undo system directly
            self.store_undo_state()  # Store state after modifications
            self.stroke_buffer.clear()
    
    def modal(self, context, event):
        # Always update mouse position
        self.last_mouse_region_x = event.mouse_region_x
        self.last_mouse_region_y = event.mouse_region_y
        
        # Handle navigation events
        if event.alt:
            context.window.cursor_set('DEFAULT')
            return {'PASS_THROUGH'}
            
        # Pass through navigation events
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 
                         'NUMPAD_PERIOD', 'HOME', 'NUMPAD_1', 'NUMPAD_2', 
                         'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_7', 
                         'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_5'}:
            context.window.cursor_set('DEFAULT')
            return {'PASS_THROUGH'}
            
        # Handle ESC to properly deactivate the addon
        if event.type == 'ESC':
            self.cleanup_handlers()
            context.window.cursor_set('DEFAULT')
            context.scene.fcurve_smooth_brush.is_active = False  # Properly deactivate the addon
            return {'CANCELLED'}
        
        if not context.scene.fcurve_smooth_brush.is_active:
            self.cleanup_handlers()
            context.window.cursor_set('DEFAULT')
            return {'CANCELLED'}
        
        # Handle custom undo/redo
        if event.type in {'Z', 'Y'} and event.ctrl:
            if len(self.undo_states) > 1:  # Need at least 2 states for undo
                if event.type == 'Z':
                    if event.shift:  # Redo
                        state = self.undo_states.pop()
                        self.restore_state(state)
                    else:  # Undo
                        self.undo_states.pop()  # Remove current state
                        if self.undo_states:
                            state = self.undo_states[-1]
                            self.restore_state(state)
                elif event.type == 'Y':  # Redo
                    state = self.undo_states.pop()
                    self.restore_state(state)
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        # Check UI regions
        for region in context.area.regions:
            if region.type in {'UI', 'TOOLS', 'HEADER', 'CHANNELS', 'HUD'}:
                if (event.mouse_x >= region.x and 
                    event.mouse_x < region.x + region.width and
                    event.mouse_y >= region.y and 
                    event.mouse_y < region.y + region.height):
                    context.window.cursor_set('DEFAULT')
                    return {'PASS_THROUGH'}
        
        context.window.cursor_set('NONE')  # Set cursor to none when over valid area
        
        if event.type == 'MOUSEMOVE':
            self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
            context.area.tag_redraw()
            
            if self.is_painting:
                current_time = time.time()
                if current_time - self.last_process_time >= self.process_interval:
                    self.smooth_curves(context)
                    self.last_process_time = current_time
        
        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.is_painting = True
                self.begin_stroke()  # Initialize stroke properly
                self.last_process_time = time.time()
            elif event.value == 'RELEASE':
                if self.is_painting:
                    self.is_painting = False
                    self.end_stroke(context)  # End stroke properly
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        if context.area.type == 'GRAPH_EDITOR':
            # Clean up any existing handlers first
            self.cleanup_handlers()
            self.clear_cache()
            
            if not context.scene.fcurve_smooth_brush.is_active:
                context.scene.fcurve_smooth_brush.is_active = True
                self.mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
                
                # Add draw handlers
                args = (self, context)
                self._handle = bpy.types.SpaceGraphEditor.draw_handler_add(
                    draw_brush_callback_px, args, 'WINDOW', 'POST_PIXEL')
                self._cursor_handle = bpy.types.SpaceGraphEditor.draw_handler_add(
                    draw_brush_cursor, (self, context, event), 'WINDOW', 'POST_PIXEL')
                
                context.window_manager.modal_handler_add(self)
                return {'RUNNING_MODAL'}
                
        return {'CANCELLED'}
    
    def __del__(self):
        """Cleanup when operator is deleted"""
        self.cleanup_handlers()
        self.clear_cache()

    def find_parent_fcurve(self, keyframe, fcurve_cache={}):
        # Use object ID as cache key
        cache_key = id(keyframe)
        if cache_key in fcurve_cache:
            return fcurve_cache[cache_key]
        
        action = keyframe.id_data
        if not hasattr(action, 'fcurves'):
            return None
        
        kf_time = keyframe.co[0]
        kf_value = keyframe.co[1]
        
        for fc in action.fcurves:
            for kf in fc.keyframe_points:
                if (abs(kf.co[0] - kf_time) < 0.0001 and 
                    abs(kf.co[1] - kf_value) < 0.0001):
                    fcurve_cache[cache_key] = fc
                    return fc
        return None
        
    def relative_smooth_keyframe(self, keyframe, factor, fcurve):
        """Smooth keyframe while preserving the overall shape"""
        original_value = self.stroke_start_values.get(fcurve, {}).get(keyframe)
        if not original_value:
            return keyframe.co[1]

        kf_index = list(fcurve.keyframe_points).index(keyframe)
        if 0 < kf_index < len(fcurve.keyframe_points) - 1:
            # Get neighboring keyframes
            prev_kf = fcurve.keyframe_points[kf_index - 1]
            next_kf = fcurve.keyframe_points[kf_index + 1]
            
            # Calculate the original relationships
            orig_prev = self.stroke_start_values[fcurve][prev_kf]
            orig_next = self.stroke_start_values[fcurve][next_kf]
            orig_current = original_value
            
            # Calculate the original relative position
            orig_range = orig_next[1] - orig_prev[1]
            if abs(orig_range) < 0.0001:  # Avoid division by zero
                return keyframe.co[1]
                
            orig_relative_pos = (orig_current[1] - orig_prev[1]) / orig_range
            
            # Calculate current smooth position
            current_prev = prev_kf.co[1]
            current_next = next_kf.co[1]
            smooth_value = (current_prev + keyframe.co[1] + current_next) / 3
            
            # Calculate new value maintaining relative position
            current_range = current_next - current_prev
            target_value = current_prev + (orig_relative_pos * current_range)
            
            # Blend between smoothed and relative-preserved value
            final_value = (smooth_value * factor) + (target_value * (1 - factor))
            return final_value
            
        return keyframe.co[1]

    def process_stroke(self, context, keyframe, factor, mode, fcurve=None):
        if mode == 'RELATIVE':
            return self.relative_smooth_keyframe(keyframe, factor, fcurve)
        elif mode == 'SMOOTH':
            return self.smooth_keyframe(keyframe, factor)
        elif mode == 'NOISE':
            return self.add_noise(keyframe, factor)
        elif mode == 'FLATTEN':
            return self.flatten_keyframe(keyframe, factor)
        elif mode == 'SHARPEN':
            return self.sharpen_keyframe(keyframe, factor)
        elif mode == 'RELAX':
            return self.relax_keyframe(keyframe, factor)
        return keyframe.co[1]

    def smooth_keyframe(self, keyframe, factor):
        fcurve = self.find_parent_fcurve(keyframe)
        if fcurve:
            kf_time = keyframe.co[0]
            kf_value = keyframe.co[1]
            
            for i, kf in enumerate(fcurve.keyframe_points):
                if (abs(kf.co[0] - kf_time) < 0.0001 and 
                    abs(kf.co[1] - kf_value) < 0.0001):
                    if i > 0 and i < len(fcurve.keyframe_points) - 1:
                        prev_value = fcurve.keyframe_points[i-1].co[1]
                        next_value = fcurve.keyframe_points[i+1].co[1]
                        smoothed = (prev_value + kf_value + next_value) / 3
                        return kf_value + (smoothed - kf_value) * factor
                    break
        return keyframe.co[1]

    def add_noise(self, keyframe, factor):
        fcurve = self.find_parent_fcurve(keyframe)
        if fcurve:
            noise = (np.random.random() - 0.5) * 2 * factor
            return keyframe.co[1] + noise
        return keyframe.co[1]

    def flatten_keyframe(self, keyframe, factor):
        fcurve = self.find_parent_fcurve(keyframe)
        if fcurve and len(fcurve.keyframe_points) > 0:
            avg = sum(k.co[1] for k in fcurve.keyframe_points) / len(fcurve.keyframe_points)
            return keyframe.co[1] + (avg - keyframe.co[1]) * factor
        return keyframe.co[1]

    def sharpen_keyframe(self, keyframe, factor):
        fcurve = self.find_parent_fcurve(keyframe)
        if fcurve:
            kf_time = keyframe.co[0]
            kf_value = keyframe.co[1]
            
            for i, kf in enumerate(fcurve.keyframe_points):
                if (abs(kf.co[0] - kf_time) < 0.0001 and 
                    abs(kf.co[1] - kf_value) < 0.0001):
                    if i > 0 and i < len(fcurve.keyframe_points) - 1:
                        prev_value = fcurve.keyframe_points[i-1].co[1]
                        next_value = fcurve.keyframe_points[i+1].co[1]
                        avg = (prev_value + next_value) / 2
                        diff = kf_value - avg
                        return kf_value + diff * factor
                    break
        return keyframe.co[1]

    def relax_keyframe(self, keyframe, factor):
        fcurve = self.find_parent_fcurve(keyframe)
        if fcurve:
            kf_time = keyframe.co[0]
            kf_value = keyframe.co[1]
            
            for i, kf in enumerate(fcurve.keyframe_points):
                if (abs(kf.co[0] - kf_time) < 0.0001 and 
                    abs(kf.co[1] - kf_value) < 0.0001):
                    if i > 0 and i < len(fcurve.keyframe_points) - 1:
                        prev_frame = fcurve.keyframe_points[i-1].co[0]
                        next_frame = fcurve.keyframe_points[i+1].co[0]
                        ideal_frame = (prev_frame + next_frame) / 2
                        frame_diff = ideal_frame - kf_time
                        keyframe.co[0] += frame_diff * factor
                    break
        return keyframe.co[1]

    def smooth_curves(self, context):
        """Main function to process and smooth the curves under the brush"""
        if not self.active_stroke:
            self.begin_stroke()
        
        # Cache commonly accessed properties
        props = context.scene.fcurve_smooth_brush
        brush_size = props.brush_size
        strength = props.strength
        mode = props.brush_mode
        
        # Get mouse position in view space
        region = context.region
        view = region.view2d
        mouse_frame, mouse_value = view.region_to_view(self.mouse_pos.x, self.mouse_pos.y)
        
        # Process each selected curve
        for fcurve in context.selected_editable_fcurves:
            if fcurve.hide:
                continue
            
            # Quick bounds check
            frame_min = min(k.co[0] for k in fcurve.keyframe_points)
            frame_max = max(k.co[0] for k in fcurve.keyframe_points)
            
            if abs(mouse_frame - frame_min) > brush_size and abs(mouse_frame - frame_max) > brush_size:
                continue  # Skip if curve is far from brush
            
            # Get keyframes to process
            all_keyframes = list(fcurve.keyframe_points)
            if props.use_acceleration:
                sample_rate = max(1, int(props.sample_rate))
                keyframes = all_keyframes[::sample_rate]
            else:
                keyframes = all_keyframes
            
            if props.affect_selected:
                keyframes = [k for k in keyframes if k.select_control_point]
            
            # Process each keyframe
            for keyframe in keyframes:
                # Convert to screen space for distance check
                screen_x, screen_y = view.view_to_region(keyframe.co[0], keyframe.co[1])
                mouse_screen_x = self.mouse_pos.x
                mouse_screen_y = self.mouse_pos.y
                
                # Calculate distance and factor
                dx = screen_x - mouse_screen_x
                dy = screen_y - mouse_screen_y
                distance = ((dx * dx + dy * dy) ** 0.5) / brush_size
                
                if distance <= 1.0:
                    # Smooth falloff
                    factor = (1 - distance ** 2) * strength
                    
                    # Apply the brush effect
                    new_value = self.process_stroke(context, keyframe, factor, mode, fcurve)
                    keyframe.co[1] = new_value
                    
                    if props.select_while_painting:
                        keyframe.select_control_point = True
            
            fcurve.update()
        
        if props.auto_frame:
            bpy.ops.graph.view_selected()

    def draw_brush_cursor(self, context, event):
        props = context.scene.fcurve_smooth_brush
        radius = props.brush_size
        
        # Single cursor circle
        segments = 16
        center = (self.last_mouse_region_x, self.last_mouse_region_y)
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        
        # Create circle vertices
        vertices = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            vertices.append((x, y))
        
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.8))
        batch.draw(shader)

    def draw_brush_callback_px(self, context):
        if not self.is_painting:
            return
            
        props = context.scene.fcurve_smooth_brush
        radius = props.brush_size
        strength = props.strength
        segments = 32
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        
        # Draw brush circle with gradient
        center = (self.mouse_pos.x, self.mouse_pos.y)
        vertices = []
        indices = []
        
        vertices.append(center)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            vertices.append((x, y))
            
            if i < segments - 1:
                indices.extend([0, i + 1, i + 2])
            else:
                indices.extend([0, segments, 1])
        
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
        
        # Simple gradient
        for i in range(4):
            alpha = 0.15 * (1 - i/4) * strength
            shader.uniform_float("color", (1.0, 1.0, 1.0, alpha))
            batch.draw(shader)

class FCurveSmoothBrushPanel(bpy.types.Panel):
    bl_label = "FCurve Smooth Brush"
    bl_idname = "GRAPH_PT_fcurve_smooth_brush"
    bl_space_type = 'GRAPH_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.fcurve_smooth_brush
        
        # Main brush controls
        row = layout.row(align=True)
        if not props.is_active:
            row.operator("graph.fcurve_smooth_brush", text="Start Brush", icon='BRUSH_DATA')
        else:
            row.label(text="Brush Active", icon='RADIOBUT_ON')
            row.prop(props, "is_active", text="", icon='X')
        
        # Brush settings
        col = layout.column(align=True)
        col.prop(props, "brush_mode", text="Mode")
        col.prop(props, "brush_size", text="Size")
        col.prop(props, "strength", text="Strength")
        col.prop(props, "iterations", text="Iterations")
        
        # Selection options
        box = layout.box()
        col = box.column()
        col.prop(props, "affect_selected", text="Selected Only")
        col.prop(props, "select_while_painting", text="Auto-Select")
        
        # Advanced options
        box = layout.box()
        box.label(text="Advanced")
        col = box.column()
        col.prop(props, "use_acceleration", text="Use Acceleration")
        if props.use_acceleration:
            col.prop(props, "sample_rate", text="Sample Rate")
        col.prop(props, "auto_frame", text="Auto Frame")
        col.prop(props, "preserve_handles", text="Preserve Handles")

# Registration
classes = (
    FCurveSmoothBrushProperties,
    FCurveSmoothBrushOperator,
    FCurveSmoothBrushPanel
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fcurve_smooth_brush = bpy.props.PointerProperty(type=FCurveSmoothBrushProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.fcurve_smooth_brush

if __name__ == "__main__":
    register()
