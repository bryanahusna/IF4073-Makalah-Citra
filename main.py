from dist_yolo import DistYOLO

from PIL import Image

if __name__ == '__main__':
    print('Dist-YOLO')
    # # Untuk Pelatihan
    # dy = DistYOLO()
    # dy.fit(1, 16, "D:/KITTI/Detection/training/image_2", "all_except_nodata_train.txt")
    
    # # Untuk Prediksi
    # dy = DistYOLO()
    # dy.load_model('trained_final.h5')
    # img = Image.open("D:/KITTI/Detection/training/image_2/000010.png")
    # result = dy.detect_image(img)
    # result.show()
