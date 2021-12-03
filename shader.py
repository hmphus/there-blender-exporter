import bpy


@bpy.app.handlers.persistent
def there_node_group_handler(dummy):
    print('Node group handler')
    node_group = bpy.data.node_groups.get('There BSDF')
    if node_group is None:
        node_group = bpy.data.node_groups.new(type='ShaderNodeTree', name='There BSDF')
    else:
        node_group.links.clear()
        node_group.nodes.clear()
    socket_color = node_group.inputs.get('Color')
    if socket_color is None:
        socket_color = node_group.inputs.new(type='NodeSocketColor', name='Color')
        socket_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_emission = node_group.inputs.get('Emission')
    if socket_emission is None:
        socket_emission = node_group.inputs.new(type='NodeSocketColor', name='Emission')
        socket_emission.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_alpha = node_group.inputs.get('Alpha')
    if socket_alpha is None:
        socket_alpha = node_group.inputs.new(type='NodeSocketFloatFactor', name='Alpha')
        socket_alpha.min_value = 0.0
        socket_alpha.max_value = 1.0
        socket_alpha.default_value = 1.0
    socket_detail_color = node_group.inputs.get('Detail Color')
    if socket_detail_color is None:
        socket_detail_color = node_group.inputs.new(type='NodeSocketColor', name='Detail Color')
        socket_detail_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_detail_alpha = node_group.inputs.get('Detail Alpha')
    if socket_detail_alpha is None:
        socket_detail_alpha = node_group.inputs.new(type='NodeSocketFloatFactor', name='Detail Alpha')
        socket_detail_alpha.min_value = 0.0
        socket_detail_alpha.max_value = 1.0
        socket_detail_alpha.default_value = 0.0
    socket_lighting = node_group.inputs.get('Lighting')
    if socket_lighting is None:
        socket_lighting = node_group.inputs.new(type='NodeSocketColor', name='Lighting')
        socket_lighting.default_value = (0.8, 0.8, 0.8, 1.0)
    socket_normal = node_group.inputs.get('Normal')
    if socket_normal is None:
        socket_normal = node_group.inputs.new(type='NodeSocketColor', name='Normal')
        socket_normal.default_value = (0.5, 0.5, 1.0, 1.0)
    socket_gloss_color = node_group.inputs.get('Gloss Color')
    if socket_gloss_color is None:
        socket_gloss_color = node_group.inputs.new(type='NodeSocketColor', name='Gloss Color')
        socket_gloss_color.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_gloss_alpha = node_group.inputs.get('Gloss Alpha')
    if socket_gloss_alpha is None:
        socket_gloss_alpha = node_group.inputs.new(type='NodeSocketFloatFactor', name='Gloss Alpha')
        socket_gloss_alpha.min_value = 0.0
        socket_gloss_alpha.max_value = 1.0
        socket_gloss_alpha.default_value = 0.0
    socket_specular_power = node_group.inputs.get('Specular Power')
    if socket_specular_power is None:
        socket_specular_power = node_group.inputs.new(type='NodeSocketFloat', name='Specular Power')
        socket_specular_power.min_value = 0.0
        socket_specular_power.max_value = 511.0
        socket_specular_power.default_value = 40.0
    socket_specular = node_group.inputs.get('Specular')
    if socket_specular is None:
        socket_specular = node_group.inputs.new(type='NodeSocketColor', name='Specular')
        socket_specular.default_value = (0.0144, 0.0144, 0.0144, 1.0)
    socket_environment = node_group.inputs.get('Environment')
    if socket_environment is None:
        socket_environment = node_group.inputs.new(type='NodeSocketColor', name='Environment')
        socket_environment.default_value = (0.0, 0.0, 0.0, 1.0)
    socket_bsdf = node_group.outputs.get('BSDF')
    if socket_bsdf is None:
        socket_bsdf = node_group.outputs.new(type='NodeSocketShader', name='BSDF')
    node_input = node_group.nodes.new(type='NodeGroupInput')
    node_input.location = (-719.8, -50.2)
    node_output = node_group.nodes.new(type='NodeGroupOutput')
    node_output.location = (583.6, 473.3)
    node_principled = node_group.nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled.location = (2.2, 321.0)
    node_diffuse = node_group.nodes.new(type='ShaderNodeBsdfDiffuse')
    node_diffuse.location = (93.1, -349.2)
    node_normal = node_group.nodes.new(type='ShaderNodeNormalMap')
    node_normal.location = (-360.2, -324.3)
    node_mix_shader = node_group.nodes.new(type='ShaderNodeMixShader')
    node_mix_shader.location = (334.8, 473.7)
    node_mix_rgb = node_group.nodes.new(type='ShaderNodeMixRGB')
    node_mix_rgb.location = (-183.8, 17.8)
    node_mix_rgb.blend_type = 'MULTIPLY'
    node_mix_rgb.inputs['Fac'].default_value = 1.0
    node_group.links.new(node_input.outputs['Color'], node_principled.inputs['Base Color'])
    node_group.links.new(node_input.outputs['Emission'], node_mix_rgb.inputs['Color1'])
    node_group.links.new(node_input.outputs['Alpha'], node_principled.inputs['Alpha'])
    node_group.links.new(node_input.outputs['Detail Color'], node_diffuse.inputs['Color'])
    node_group.links.new(node_input.outputs['Detail Alpha'], node_mix_shader.inputs['Fac'])
    node_group.links.new(node_input.outputs['Lighting'], node_mix_rgb.inputs['Color2'])
    node_group.links.new(node_input.outputs['Normal'], node_normal.inputs['Color'])
    node_group.links.new(node_mix_rgb.outputs['Color'], node_principled.inputs['Emission'])
    node_group.links.new(node_normal.outputs['Normal'], node_principled.inputs['Normal'])
    node_group.links.new(node_normal.outputs['Normal'], node_diffuse.inputs['Normal'])
    node_group.links.new(node_principled.outputs['BSDF'], node_mix_shader.inputs[1])
    node_group.links.new(node_diffuse.outputs['BSDF'], node_mix_shader.inputs[2])
    node_group.links.new(node_mix_shader.outputs['Shader'], node_output.inputs['BSDF'])


def register_shader():
    bpy.app.handlers.load_post.append(there_node_group_handler)


def unregister_shader():
    bpy.app.handlers.load_post.remove(there_node_group_handler)