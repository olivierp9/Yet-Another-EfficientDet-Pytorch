from gen_utils import *

if __name__ == "__main__":

    scene_renderer = SceneRender(128, normal_map=True)

    # scene_renderer.add_bunny_mesh(np.array([0, 0, 0]), np.array([0, 0, 0]))
    # scene_renderer.draw_mesh()
    # scene_renderer.render(np.array([0, 1, 0]), np.array([0, 0, 1]))  # hopes nothing happens if pos is like up
    # img, _ = scene_renderer.get_image_and_bbox()
    # plt.imshow(img)
    # plt.show()
    import gzip


    img = scene_renderer.generate_ae_views(0.7)
    #
    f = gzip.GzipFile("full.npy.gz", "w")
    np.save(file=f, arr=img)
    f.close()
    #
    # # f = gzip.GzipFile('my_array.npy.gz', "r")
    # # img = np.load(f)
    #
    # print(np.min(img))
    # print(np.max(img))
    # print(np.mean(img))
    # print(np.std(img))
    # scene_renderer.generate_examples()



