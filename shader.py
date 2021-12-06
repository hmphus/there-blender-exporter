import bpy


@bpy.app.handlers.persistent
def there_node_group_handler(dummy):
    node_tree = bpy.data.node_groups.get('There BSDF')
    if node_tree is None:
        node_tree = bpy.data.node_groups.new(type='ShaderNodeTree', name='There BSDF')
    else:
        node_tree.links.clear()
        node_tree.nodes.clear()
    socket_color = node_tree.inputs.get('Color')
    if socket_color is None:
        socket_color = node_tree.inputs.new(type='NodeSocketColor', name='Color')
        socket_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_emission = node_tree.inputs.get('Emission')
    if socket_emission is None:
        socket_emission = node_tree.inputs.new(type='NodeSocketColor', name='Emission')
        socket_emission.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_alpha = node_tree.inputs.get('Alpha')
    if socket_alpha is None:
        socket_alpha = node_tree.inputs.new(type='NodeSocketFloatFactor', name='Alpha')
        socket_alpha.min_value = 0.0
        socket_alpha.max_value = 1.0
        socket_alpha.default_value = 1.0
    socket_detail_color = node_tree.inputs.get('Detail Color')
    if socket_detail_color is None:
        socket_detail_color = node_tree.inputs.new(type='NodeSocketColor', name='Detail Color')
        socket_detail_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_detail_alpha = node_tree.inputs.get('Detail Alpha')
    if socket_detail_alpha is None:
        socket_detail_alpha = node_tree.inputs.new(type='NodeSocketFloatFactor', name='Detail Alpha')
        socket_detail_alpha.min_value = 0.0
        socket_detail_alpha.max_value = 1.0
        socket_detail_alpha.default_value = 0.0
    socket_lighting = node_tree.inputs.get('Lighting')
    if socket_lighting is None:
        socket_lighting = node_tree.inputs.new(type='NodeSocketColor', name='Lighting')
        socket_lighting.default_value = (0.8, 0.8, 0.8, 1.0)
    socket_normal = node_tree.inputs.get('Normal')
    if socket_normal is None:
        socket_normal = node_tree.inputs.new(type='NodeSocketColor', name='Normal')
        socket_normal.default_value = (0.5, 0.5, 1.0, 1.0)
    socket_gloss_color = node_tree.inputs.get('Gloss Color')
    if socket_gloss_color is None:
        socket_gloss_color = node_tree.inputs.new(type='NodeSocketColor', name='Gloss Color')
        socket_gloss_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_gloss_alpha = node_tree.inputs.get('Gloss Alpha')
    if socket_gloss_alpha is None:
        socket_gloss_alpha = node_tree.inputs.new(type='NodeSocketFloatFactor', name='Gloss Alpha')
        socket_gloss_alpha.min_value = 0.0
        socket_gloss_alpha.max_value = 1.0
        socket_gloss_alpha.default_value = 0.0
    socket_specular_power = node_tree.inputs.get('Specular Power')
    if socket_specular_power is None:
        socket_specular_power = node_tree.inputs.new(type='NodeSocketFloat', name='Specular Power')
        socket_specular_power.min_value = 0.0
        socket_specular_power.max_value = 511.0
        socket_specular_power.default_value = 40.0
    socket_specular = node_tree.inputs.get('Specular')
    if socket_specular is None:
        socket_specular = node_tree.inputs.new(type='NodeSocketColor', name='Specular')
        socket_specular.default_value = (0.1255, 0.1255, 0.1255, 1.0)
    socket_environment = node_tree.inputs.get('Environment')
    if socket_environment is None:
        socket_environment = node_tree.inputs.new(type='NodeSocketColor', name='Environment')
        socket_environment.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_bsdf = node_tree.outputs.get('BSDF')
    if socket_bsdf is None:
        socket_bsdf = node_tree.outputs.new(type='NodeSocketShader', name='BSDF')
    node_input = node_tree.nodes.new(type='NodeGroupInput')
    node_input.location = (-808, -41)
    node_vertex = node_tree.nodes.new(type='ShaderNodeVertexColor')
    node_vertex.location = (-808, -418)
    node_principled = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled.location = (54, 346)
    node_diffuse = node_tree.nodes.new(type='ShaderNodeBsdfDiffuse')
    node_diffuse.location = (144, -352)
    node_normal = node_tree.nodes.new(type='ShaderNodeNormalMap')
    node_normal.location = (-221, -239)
    node_mix_shader = node_tree.nodes.new(type='ShaderNodeMixShader')
    node_mix_shader.location = (452, 477)
    node_mix_rgb_color = node_tree.nodes.new(type='ShaderNodeMixRGB')
    node_mix_rgb_color.location = (-492, 373)
    node_mix_rgb_color.blend_type = 'MULTIPLY'
    node_mix_rgb_color.inputs['Fac'].default_value = 1.0
    node_mix_rgb_emission = node_tree.nodes.new(type='ShaderNodeMixRGB')
    node_mix_rgb_emission.location = (-492, 81)
    node_mix_rgb_emission.blend_type = 'MULTIPLY'
    node_mix_rgb_emission.inputs['Fac'].default_value = 1.0
    node_mix_rgb_detail = node_tree.nodes.new(type='ShaderNodeMixRGB')
    node_mix_rgb_detail.location = (-492, -378)
    node_mix_rgb_detail.blend_type = 'MULTIPLY'
    node_mix_rgb_detail.inputs['Fac'].default_value = 1.0
    node_mix_rgb_lighting = node_tree.nodes.new(type='ShaderNodeMixRGB')
    node_mix_rgb_lighting.location = (-211, 5)
    node_mix_rgb_lighting.blend_type = 'MULTIPLY'
    node_mix_rgb_lighting.inputs['Fac'].default_value = 1.0
    node_output = node_tree.nodes.new(type='NodeGroupOutput')
    node_output.location = (673, 473)
    node_tree.links.new(node_input.outputs['Color'], node_mix_rgb_color.inputs['Color1'])
    node_tree.links.new(node_input.outputs['Emission'], node_mix_rgb_emission.inputs['Color1'])
    node_tree.links.new(node_input.outputs['Alpha'], node_principled.inputs['Alpha'])
    node_tree.links.new(node_input.outputs['Detail Color'], node_mix_rgb_detail.inputs['Color1'])
    node_tree.links.new(node_input.outputs['Detail Alpha'], node_mix_shader.inputs['Fac'])
    node_tree.links.new(node_input.outputs['Lighting'], node_mix_rgb_lighting.inputs['Color2'])
    node_tree.links.new(node_input.outputs['Normal'], node_normal.inputs['Color'])
    node_tree.links.new(node_vertex.outputs['Color'], node_mix_rgb_color.inputs['Color2'])
    node_tree.links.new(node_vertex.outputs['Color'], node_mix_rgb_emission.inputs['Color2'])
    node_tree.links.new(node_vertex.outputs['Color'], node_mix_rgb_detail.inputs['Color2'])
    node_tree.links.new(node_mix_rgb_color.outputs['Color'], node_principled.inputs['Base Color'])
    node_tree.links.new(node_mix_rgb_emission.outputs['Color'], node_mix_rgb_lighting.inputs['Color1'])
    node_tree.links.new(node_mix_rgb_detail.outputs['Color'], node_diffuse.inputs['Color'])
    node_tree.links.new(node_mix_rgb_lighting.outputs['Color'], node_principled.inputs['Emission'])
    node_tree.links.new(node_normal.outputs['Normal'], node_principled.inputs['Normal'])
    node_tree.links.new(node_normal.outputs['Normal'], node_diffuse.inputs['Normal'])
    node_tree.links.new(node_principled.outputs['BSDF'], node_mix_shader.inputs[1])
    node_tree.links.new(node_diffuse.outputs['BSDF'], node_mix_shader.inputs[2])
    node_tree.links.new(node_mix_shader.outputs['Shader'], node_output.inputs['BSDF'])


def register_shader():
    bpy.app.handlers.load_post.append(there_node_group_handler)


def unregister_shader():
    bpy.app.handlers.load_post.remove(there_node_group_handler)