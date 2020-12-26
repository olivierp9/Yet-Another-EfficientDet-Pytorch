import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
import time
from math import sin, cos, pi
import open3d as o3d
import cv2
from view_sampler import hinter_sampling
from math import pi
import datetime
import uuid
from pycococreatortools import pycococreatortools
import matplotlib.pyplot as plt
import json


def opengl_error_check():
    error = glGetError()
    if error != GL_NO_ERROR:
        print("OPENGL_ERROR: ", error)


class Shaders:
    @staticmethod
    def vertex_src() -> str:
        return """
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

    @staticmethod
    def fragment_src() -> str:
        return """
        # version 330
        in vec2 v_texture;
        out vec4 out_color;
        float near = 0.1;
        float far = 2;
        
        float LinearizeDepth(float depth) 
        {
            return near * far / (far + depth * (near - far));
            // the good one is linear in distance
            // return (depth-near)/(far-near);
            // or maybe only return the real depth? does not change a lot because we normalize
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
float far = 10;

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

MESH_PATH = "/home/olivier/Desktop/shape_net/meshes.txt"

# TODO add sampling for random viewpoint


class SceneRender:
    def __init__(self, size: int):
        # TODO add item to identify as input
        self.size = size
        self.vertices = np.array([])
        self.indices = np.array([])

        # initializing glfw library
        if not glfw.init():
            raise Exception("glfw can not be initialized!")
        opengl_error_check()
        # glfw.window_hint(glfw.VISIBLE, False)
        # creating the window
        window = glfw.create_window(size, size, "My OpenGL window", None, None)
        opengl_error_check()
        self.window = window
        # check if window was created
        if not window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # set window's position
        glfw.set_window_pos(window, 0, 0)
        opengl_error_check()
        with open(MESH_PATH) as f:
            content = f.readlines()

        self.meshes = [x.strip() for x in content]
        self.meshes_len = len(self.meshes)

        def window_resize(window, width: int, height: int) -> None:
            glViewport(0, 0, width, height)
            projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 10)
            glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)

        # set the callback function for window resize
        glfw.set_window_size_callback(window, window_resize)
        opengl_error_check()
        # make the context current
        glfw.make_context_current(window)
        opengl_error_check()
        print(Shaders.vertex_src())
        print(Shaders.fragment_src())

        self.shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        opengl_error_check()
    # def add_mesh(self, mesh_path: str, pose: np.ndarray, rotation: np.ndarray):

    def add_bunny_mesh(self, position, rotation):
        mesh = o3d.io.read_triangle_mesh("mesh/mini_bunny.obj")
        R = mesh.get_rotation_matrix_from_xyz(rotation)
        mesh.rotate(R, center=(0, 0, 0))
        mesh.translate(position)

        verts = np.asarray(mesh.vertices)
        max_idx = 0
        if len(self.indices)>0:
            max_idx = np.max(self.indices)+1
        triangs = np.asarray(mesh.triangles)+max_idx

        self.vertices = np.append(self.vertices, np.array(verts.flatten(), dtype=np.float32))
        self.indices = np.append(self.indices, np.array(triangs.flatten(), dtype=np.uint32))
        return

    def add_random_mesh(self, radius=1):
        mesh_path = self.meshes[np.random.randint(0, self.meshes_len)]
        mesh = o3d.io.read_triangle_mesh(mesh_path.replace("model_normalized_vhacd.obj", "model_normalized.obj"))
        pos = np.random.uniform(0,1,3)
        pos = pos/np.linalg.norm(pos)
        mesh.translate(pos)
        # mesh = o3d.io.read_triangle_mesh("mesh/mini_bunny.obj")
        verts = np.asarray(mesh.vertices)

        # print(len(self.vertices))
        # print(len(self.indices))
        # print(np.max(self.indices))

        max_idx = 0
        if len(self.indices)>0:
            max_idx = np.max(self.indices)+1
        triangs = np.asarray(mesh.triangles)+max_idx

        self.vertices = np.append(self.vertices, verts.flatten())
        self.indices = np.append(self.indices, triangs.flatten())

    def clear_meshes(self):
        self.vertices = np.array([], dtype=np.float32)
        self.indices = np.array([], dtype=np.uint32)

    def draw_mesh(self):
        # Vertex Buffer Object
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        opengl_error_check()
        # Element Buffer Object
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        opengl_error_check()
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 3, ctypes.c_void_p(0))
        opengl_error_check()
        glUseProgram(self.shader)
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        opengl_error_check()
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")

        opengl_error_check()

    def render(self, position: np.ndarray, up: np.ndarray, out_size=0, debug=False) -> None:

        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = pyrr.matrix44.create_look_at(pyrr.Vector3(position), pyrr.Vector3([0.0, 0.0, 0.0]),
                                            pyrr.Vector3(up))
        opengl_error_check()

        projection = pyrr.matrix44.create_perspective_projection_matrix(45, 720 / 720, 0.1, 10)
        translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, translation)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)

        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

    @staticmethod
    def get_ups_from_point(normal: np.ndarray, increment: float):
        # https://stackoverflow.com/questions/27714014/3d-point-on-circumference-of-a-circle-with-a-center-radius-and-normal-vector
        s = 1.0 / np.sqrt(normal.dot(normal))
        v3x = s * normal[0]
        v3y = s * normal[1]
        v3z = s * normal[2]

        if v3x * v3x + v3z * v3z == 0:
            a = 0

        s = 1.0 / np.sqrt(v3x * v3x + v3z * v3z)
        v1x = s * v3z
        v1y = 0.0
        v1z = -s * v3x

        v2x = v3y * v1z - v3z * v1y
        v2y = v3z * v1x - v3x * v1z
        v2z = v3x * v1y - v3y * v1x

        list_of_up = []
        num_inter = int(2 * pi / increment)
        for i in range(0, num_inter):
            a = i * increment
            px = 1 * (v1x * cos(a) + v2x * sin(a))
            py = 1 * (v1y * cos(a) + v2y * sin(a))
            pz = 1 * (v1z * cos(a) + v2z * sin(a))
            list_of_up.append([px, py, pz])

        return np.array(list_of_up)

    @staticmethod
    def generate_transform(radius: float=1.0):
        pos = np.random.uniform(0, 1, 3)
        pos *= radius / np.linalg.norm(pos)

        rot = np.random.uniform(0, 2 * pi)
        return pos, rot

    def get_bbox(self, img):
        tmp_img = np.array(img, copy=True)
        tmp_img[tmp_img > 0] = 1

        rows_with_white = np.max(tmp_img, axis=1)
        cols_with_white = np.max(tmp_img, axis=0)

        row_low = np.argmax(rows_with_white)
        row_high = self.size - np.argmax(rows_with_white[::-1])
        col_low = np.argmax(cols_with_white)
        col_high = self.size - np.argmax(cols_with_white[::-1])

        return col_low, row_low, col_high, row_high

    def get_image_and_bbox(self, debug: bool = False, boxes: bool = False):
        out = glReadPixels(0, 0, self.size, self.size, GL_DEPTH_COMPONENT, GL_FLOAT)

        # glfw.swap_buffers(self.window)

        rgb_flipped = np.frombuffer(out, dtype=np.float32).reshape(self.size, self.size, 1)

        min = np.min(rgb_flipped)
        max = np.max(rgb_flipped)
        if max == min:
            print("Error nothing to see!")
            return None

        rgb_flipped[rgb_flipped == 1] = 0

        tmp_flipped = rgb_flipped  # verify just in case

        if boxes:
            col_low, row_low, col_high, row_high = self.get_bbox(tmp_flipped)

            # just in case?
            # norm_image = cv2.normalize(rgb_flipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            if debug:
                image = cv2.rectangle(rgb_flipped, (col_low, row_low), (col_high, row_high), (255, 255, 255), 2)
                cv2.imshow("test", image)
                cv2.waitKey()

            return rgb_flipped, [col_low, row_low, col_high, row_high]
        else:
            return rgb_flipped, None

    def mult_generate_exemples(self, max_bunnies: int=5, number_of_views: int=10, max_number_of_random_object:int = 5,
                          threshold: float=0.25, radius: float=1.0):
        val = 50
        out_boxes = np.zeros((val*number_of_views, max_bunnies, 4))
        out_images = np.zeros((val*number_of_views, self.size, self.size, 1))

        for i in range(val):
            error = True
            while error:
                try:
                    images, boxes = scene_renderer.generate_examples(max_bunnies, number_of_views)
                    error = False
                except:
                    pass
            out_boxes[int(i*number_of_views):int((i+1)*number_of_views), :, :] = boxes
            out_images[int(i*number_of_views):int((i+1)*number_of_views), :, :, :] = images
            print(i)
        return out_images, out_boxes

    def generate_examples(self, max_bunnies: int=5, number_of_views: int=10, max_number_of_random_object:int = 5,
                          threshold: float=0.5, radius: float=1.0):

        num_bunnies = np.random.randint(1, max_bunnies+1)  # +1 because high is exclusive
        bunnies_pos = np.zeros((num_bunnies, 3))
        bunnies_rot = np.zeros((num_bunnies, 3))
        bunnies_box = np.zeros((num_bunnies, number_of_views, 4))

        num_rand_object = np.random.randint(1, max_number_of_random_object+1)
        obj_pos = np.zeros((num_rand_object, 3))
        obj_rot = np.zeros((num_rand_object, 3))

        bunny_images = np.zeros((num_bunnies, number_of_views, self.size, self.size, 1))
        out_images = np.zeros((number_of_views, self.size, self.size, 1))
        out_boxes = np.zeros((number_of_views, max_bunnies, 4))

        views = hinter_sampling(2562, 1)[0]  # type this shit

        rand_views_idx = np.random.randint(0, views.shape[0], number_of_views)

        views = views[rand_views_idx, :]
        # use fcl
        for bunny_idx in range(num_bunnies):
            not_far_enough = True
            while not_far_enough:
                not_far_enough = False
                bunnies_pos[bunny_idx], bunnies_rot[bunny_idx] = self.generate_transform(radius)
                for j in range(bunny_idx):
                    if np.linalg.norm(bunnies_pos[bunny_idx]-bunnies_pos[j]) < 0.3:
                        not_far_enough = True

        for obj_idx in range(num_rand_object):
            obj_pos[obj_idx, :], obj_rot[obj_idx, :] = self.generate_transform(radius)

        for bunny_idx in range(num_bunnies):
            self.clear_meshes()
            self.add_bunny_mesh(bunnies_pos[bunny_idx, :], bunnies_rot[bunny_idx, :])
            self.draw_mesh()
            for view_idx in range(number_of_views):
                self.render(views[view_idx, :]*3, np.array([0, 0, 1]))  # hopes nothing happens if pos is like up

                # TODO if boxes overlap do something
                bunny_images[bunny_idx, view_idx, :, :, :], bunnies_box[bunny_idx, view_idx, :] = \
                    self.get_image_and_bbox(boxes=True)
            pos = 0  # generate poses for each bunnies

        self.clear_meshes()
        for obj_idx in range(num_rand_object):
            self.add_random_mesh()  # use pose and rot

        for bunny_idx in range(num_bunnies):
            self.add_bunny_mesh(bunnies_pos[bunny_idx, :], bunnies_rot[bunny_idx, :])
        self.draw_mesh()

        for view_idx in range(number_of_views):
            self.render(views[view_idx, :] * 3, np.array([0, 0, 1]))  # hopes nothing happens if pos is like up

            out_images[view_idx, :, :], _ = self.get_image_and_bbox()

            for bunny_idx in range(num_bunnies):
                box = np.array(bunnies_box[bunny_idx, view_idx, :], dtype=np.int)
                original_bunny = bunny_images[bunny_idx, view_idx, :, :, :]
                current_bunny = out_images[view_idx, :, :, :]

                box_original_bunny = original_bunny[box[1]:box[3]+1, box[0]:box[2]+1, :]
                box_current_bunny = current_bunny[box[1]:box[3]+1, box[0]:box[2]+1, :]

                non_background = box_original_bunny > 0

                # original_bunny_non_background = box_original_bunny[non_background]
                # current_bunny_non_background = box_current_bunny[non_background]

                # sum_same_pixels = np.sum(original_bunny_non_background == current_bunny_non_background)
                # sum_non_background_pixels = np.sum(non_background)

                # percentage = sum_same_pixels/sum_non_background_pixels

                # tmp = np.copy(box_original_bunny)
                # minn = np.min(tmp[tmp>0])
                # maxx = np.max(tmp[tmp>0])
                # norm_im = np.copy(box_original_bunny)
                # norm_im[tmp>0] = (norm_im[tmp>0]-minn)/(maxx)

                # plt.imshow(norm_im, cmap='viridis')
                # plt.show()

                # tmp = np.copy(box_current_bunny)
                # minn = np.min(tmp[tmp>0])
                # maxx = np.max(tmp[tmp>0])
                # norm_im = np.copy(box_current_bunny)
                # norm_im[tmp>0] = (norm_im[tmp>0]-minn)/(maxx)

                # plt.imshow(norm_im, cmap='viridis')
                # plt.show()
                # plt.imshow(box_current_bunny, cmap='viridis')
                # plt.show()

                tmp = np.copy(box_current_bunny)
                tmp[non_background] /= box_original_bunny[non_background]
                minn = np.min(tmp[non_background])
                maxx = np.max(tmp[non_background])
                norm_im = np.copy(tmp)
                if (minn != maxx):
                    norm_im[non_background] = (norm_im[non_background]-minn)/(maxx)

                # plt.imshow(norm_im, cmap='viridis')
                # plt.show()

                (values, counts) = np.unique(norm_im[norm_im>0], return_counts=True)
                ind = np.argmax(counts)

                if counts[ind] / np.sum(box_original_bunny>0) > threshold:
                    # do something (keep the viewport for out)
                    out_boxes[view_idx, bunny_idx, :] = bunnies_box[bunny_idx, view_idx, :]

        return out_images, out_boxes


if __name__ == "__main__":
    ROOT_DIR = 'opengl/train'
    IMAGE_DIR = 'opengl/images'
    ANNOTATION_DIR = 'opengl/annotations'

    INFO = {
        "description": "Example Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "waspinator",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'bunny',
            'supercategory': 'bunny',
        }
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    scene_renderer = SceneRender(512)

    a = time.time()
    images, boxes = scene_renderer.mult_generate_exemples(5, 100)  # generate exemple but with different mesh inner and outer

    mean = np.mean(images[images>0])
    std = np.std(images[images>0])

    images[images > 0] = (images[images>0]-mean)/std
    for image, boxs in zip(images, boxes):  #not gonna work
        random_name = f"{str(uuid.uuid4())}.npy"
        image_info = pycococreatortools.create_image_info(
            image_id, random_name, image.shape)
        coco_output["images"].append(image_info)
        alo = False
        # write image to images

        # for each bunny add an annotation

        for i in range(boxs.shape[1]):
            box = boxs[i, :]
            if np.sum(box == np.zeros(4))==4:
                continue
            alo = True

            class_id = 1
            category_info = {'id': class_id}

            annotation_info = pycococreatortools.create_annotation_info(
                annotation_id, image_id, category_info, None,
                image.shape, tolerance=1, bounding_box=box)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            annotation_id += 1
        if alo:
            image_id += 1
            # image = np.array(image)
            # non_background = image>0
            # maxx = np.max(image[non_background])
            # minn = np.min(image[non_background])
            # norm_im = np.copy(image)
            # norm_im[non_background] = (image[non_background]-minn)/(maxx-minn)
            np.save(f"{IMAGE_DIR}/{random_name}", image)
    with open(f'{ROOT_DIR}/instances_train.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    b = time.time()
    print(b-a)

