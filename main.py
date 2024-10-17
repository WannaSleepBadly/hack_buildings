import tensorflow as tf
import numpy as np
import pandas as pd
from ultralytics import YOLO
import argparse


class HeightModel:
    """
    Модель предсказания высоты здания на основе фото здания, координат здания и координат камеры
    """

    def __init__(self, adr, file):
        self.adr = adr
        self.file = file

        self.img_model = YOLO('weights/best.pt')  # Модель для детекции зданий
        self.model = tf.keras.models.load_model('weights/BuildingHeights.keras')  # Модель для предсказания

    def preprocess(self, line):
        """
        Предобработка входов для модели предсказания

        Args:
            line (pandas series): Строка из csv файла с координатами и названием фото

        Returns:
            numpy array: Список формата [высота рамки детектированного здания,
            x координата здания, y координата здания,
            x координата камеры, y кордината камеры]
        """

        def get_height(result):  # Возвращает высоту детектированной рамки
            return result[0].boxes.xywh[0][3]

        def get_cords(line):  # Получение координат из строки для csv файла
            return list(map(float, line.split(',')))

        line = line.tolist()
        height = get_height(self.img_model(self.adr + line[0])).cpu().numpy()
        x_b, y_b = get_cords(line[1])
        x_c, y_c = get_cords(line[2])
        return np.array([height, x_b, y_b, x_c, y_c])

    def get_predictions(self):
        """
        Предсказание высоты здания по координатам камеры, здания и фото здания

        Returns:
            numpy array:  Предсказанния высоты для всего датасета
        """

        dataset = pd.read_csv(self.adr + self.file)  # Загрузка датасета
        data = np.array(list(map(self.preprocess, dataset.to_numpy())))  # Входы

        x = self.model(data).cpu().numpy()
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict results.")
    parser.add_argument(
        "--srcdir",
        type=str,
        help="Path to the source directory containing a csv file and photos.",
    )
    parser.add_argument(
        "--srccsv",
        type=str,
        help="Name of the source csv file.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output csv file.",
    )
    args = parser.parse_args()

    # Получение предсказаний
    model = HeightModel(args.srcdir, args.srccsv)
    result = model.get_predictions()

    # Сохранение предсказаний
    dataset = pd.read_csv(args.srcdir + args.srccsv)
    dataset['prediction'] = result
    dataset.to_csv(args.dst)
