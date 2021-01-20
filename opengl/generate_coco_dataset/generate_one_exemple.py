from gen_utils import *

if __name__ == "__main__":

    scene_renderer = SceneRender(512)

    depth, normals = scene_renderer.generate_one_test_example()

    np.save("../depth_map_test.npy", depth)
    np.save("../depth_normal_test.npy", normals)

    depth[depth>0] = (depth[depth>0]-np.min(depth[depth>0]))/(np.max(depth[depth>0])-np.min(depth[depth>0]))
    plt.imshow(depth)
    plt.show()

    plt.imshow(normals)
    plt.show()