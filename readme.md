
##### В задании было проделано:

1) EDA и обработка признаков
2) визуализация данных теста и трейна с использованием диаграммы рассеяния
3) обучение линейной регрессии только на вещественных признаках, и с добавлением категориальных
4) использовался GridSearchCV для подбора оптимальных параметра
5) применялась L1 регуляризация
6) для модели линейной регрессии реализован сервис на FsatAPI:
	- на вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины
	- на вход подается csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом — предсказаниями на этих объектах
