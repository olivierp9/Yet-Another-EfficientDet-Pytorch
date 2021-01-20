import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
import math
import time
from math import sin, cos, pi
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


def opengl_error_check():
    error = glGetError()
    if error != GL_NO_ERROR:
        print("OPENGL_ERROR: ", error)


vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
out vec3 v_color;
out vec2 v_texture;
void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330
in vec2 v_texture;
out vec4 out_color;
float near = 0.1;
float far = 2;

float LinearizeDepth(float depth) 
{
    //return near * far / (far + depth * (near - far));
    return depth;
    //return depth;
    //float z = depth * 2.0 - 1.0; // back to NDC 
    //return (2.0 * near * far) / (far + near - z * (far - near));	
}

uniform sampler2D s_texture;
void main()
{
    out_color = vec4(vec3(LinearizeDepth(gl_FragCoord.z)), 1.0);
}
"""

MESH_PATH = "/home/olivier/Desktop/shape_net/meshes.txt"

# TODO add sampling for random viewpoint

def add_random_mesh(vert, ind):
    mesh_path = meshes[np.random.randint(0, meshes_len)]
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.io.read_triangle_mesh("mesh/mini_bunny.obj")
    verts = np.asarray(mesh.vertices)
    triangs = np.asarray(mesh.triangles)

    vertices = np.append(vert, np.array(verts.flatten(), dtype=np.float32))
    indices = np.append(ind, np.array(triangs.flatten(), dtype=np.uint32))
    vertices = np.array(vertices,dtype=np.float32)
    indices = np.array(indices,dtype=np.uint32)
    return vertices, indices


def window_resize(window, width: int, height: int) -> None:
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 2)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

size = 128
vertices = np.array([])
indices = np.array([])


# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# glfw.window_hint(glfw.VISIBLE, False)
# creating the window
window = glfw.create_window(size, size, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, size, size)

with open(MESH_PATH) as f:
    content = f.readlines()

meshes = [x.strip() for x in content]
meshes_len = len(meshes)

vertices, indices = add_random_mesh(vertices, indices)

# set the callback function for window resize
# glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                        compileShader(fragment_src, GL_FRAGMENT_SHADER))


# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 3, ctypes.c_void_p(0))

glUseProgram(shader)
glClearColor(0, 0, 0, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 720 / 720, 0.1, 2)
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(model_loc, 1, GL_FALSE, translation)

while True:
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = pyrr.matrix44.create_look_at(pyrr.Vector3([1,0,0]), pyrr.Vector3([0.0, 0.0, 0.0]),
                                        pyrr.Vector3([0,0,1]))

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    print(len(indices))
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    out = glReadPixels(0, 0, size, size, GL_DEPTH_COMPONENT, GL_FLOAT)
    opengl_error_check()
    glfw.swap_buffers(window)

    rgb_flipped = np.frombuffer(out, dtype=np.float32).reshape(size, size, 1)
    min = np.min(rgb_flipped)
    max = np.max(rgb_flipped)
    # print(min, max)
    if max == min:
        print("Error nothing to see!")

    # flipping nothing
    rgb_flipped[rgb_flipped == 1] = 0

    #
    tmp_flipped = rgb_flipped
    tmp_flipped[tmp_flipped > 0] = 1

    rows_with_white = np.max(tmp_flipped, axis=1)
    cols_with_white = np.max(tmp_flipped, axis=0)

    row_low = np.argmax(rows_with_white)
    row_high = size - np.argmax(rows_with_white[::-1])
    col_low = np.argmax(cols_with_white)
    col_high = size - np.argmax(cols_with_white[::-1])
    # just in case?
    norm_image = cv2.normalize(rgb_flipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if False:
        image = cv2.rectangle(norm_image, (col_low, row_low), (col_high, row_high), (255, 255, 255), 2)
        # shoud we display images?
        cv2.imshow("test", norm_image)
        cv2.waitKey()

