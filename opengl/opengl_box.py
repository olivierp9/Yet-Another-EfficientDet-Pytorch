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
    return near * far / (far + depth * (near - far));
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

size = 512

# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 2)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

glfw.window_hint(glfw.VISIBLE, False)
# creating the window
window = glfw.create_window(size, size, "My OpenGL window", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, size, size)

# set the callback function for window resize
# glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

mesh = o3d.io.read_triangle_mesh("mesh/mini_bunny.obj")
verts = np.asarray(mesh.vertices)
triangs = np.asarray(mesh.triangles)

vertices = np.array(verts.flatten(), dtype=np.float32)
indices = np.array(triangs.flatten(), dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

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

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 720/720, 0.1, 2)
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))

model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(model_loc, 1, GL_FALSE, translation)
# glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

# the main application loop
start = time.time()
c = 0
from view_sampler import sample_views
from view_sampler import hinter_sampling
import warnings
# views, level = sample_views(2562, 1)
views = hinter_sampling(2562, 1)[0]
num_views = len(views)
c_views = 0

# while not glfw.window_should_close(window):


def get_ups_from_point(normal: np.ndarray, increment: float):
    # https://stackoverflow.com/questions/27714014/3d-point-on-circumference-of-a-circle-with-a-center-radius-and-normal-vector
    s = 1.0/np.sqrt(normal.dot(normal))
    v3x = s*normal[0]
    v3y = s*normal[1]
    v3z = s*normal[2]

    if v3x*v3x+v3z*v3z == 0:
        a = 0

    s = 1.0/np.sqrt(v3x*v3x+v3z*v3z)
    v1x = s*v3z
    v1y = 0.0
    v1z = -s*v3x

    v2x = v3y * v1z - v3z * v1y
    v2y = v3z * v1x - v3x * v1z
    v2z = v3x * v1y - v3y * v1x

    list_of_up = []
    num_inter = int(2*pi/increment)
    for i in range(0, num_inter):
        a = i*increment
        px = 1 * (v1x * cos(a) + v2x * sin(a))
        py = 1 * (v1y * cos(a) + v2y * sin(a))
        pz = 1 * (v1z * cos(a) + v2z * sin(a))
        list_of_up.append([px, py, pz])

    return np.array(list_of_up)

now_a = time.time()
saving_size = 128
test = np.zeros((2652*36, saving_size, saving_size, 1))
c_test = 0
views_coutner = 0
for i in range(num_views):
    pos = np.array(views[i, :])

    if i == 1385 or i == 1433:
        tmp_pos = np.array([pos[1],pos[0],pos[2]])
        ups = get_ups_from_point(-2*tmp_pos, 2*pi/36)
        for j in range(ups.shape[0]):
            ups[j,0], ups[j,1], ups[j,2] = ups[j,1], ups[j,0], ups[j,2]
    else:
        ups = get_ups_from_point(-2*pos, 2*pi/36)
    print(i)
    for up in ups:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # print(i)
        view = pyrr.matrix44.create_look_at(pyrr.Vector3(pos), pyrr.Vector3([0.0, 0.0, 0.0]),
                                            pyrr.Vector3(up))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        out = glReadPixels(0, 0, size, size,  GL_DEPTH_COMPONENT, GL_FLOAT)

        rgb_flipped = np.frombuffer(out,
                                    dtype=np.float32).reshape(size, size, 1)
        if np.min(rgb_flipped) == np.max(rgb_flipped):
            test = 0
        rgb_flipped[rgb_flipped==1] = 0
        rgb_flipped[rgb_flipped>0] = 1

        rows_with_white = np.max(rgb_flipped, axis=1)
        cols_with_white = np.max(rgb_flipped, axis=0)

        row_low = np.argmax(rows_with_white)
        row_high = size-np.argmax(rows_with_white[::-1])
        col_low = np.argmax(cols_with_white)
        col_high = size-np.argmax(cols_with_white[::-1])
        norm_image = cv2.normalize(rgb_flipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Blue color in BGR
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        # print((row_low, col_low), (row_high, col_high))
        image = cv2.rectangle(norm_image, (col_low, row_low), (col_high, row_high), color, thickness)

        cv2.imshow("test", norm_image)
        cv2.waitKey(10)


glfw.terminate()
