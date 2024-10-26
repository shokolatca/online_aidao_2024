# Авторы
https://github.com/shokolatca \\
https://github.com/pettum0104 \\
https://github.com/tymoorchik \\
Команда: looking for job \\

# Task 1

## Создание новых датасетов

В силу того, что в задаче используются 2 атласа разных размерностей, мы разбили исходный датасет на 2. Затем мы смотрели на корреляции признаков внутри каждого выделенного датасета, что дало нам нужный промежуток фич, а так же их правильный порядок. Далее мы посчитали внутри каждого выделенного датасета 8 наиболее коррелированных  с ним объектов. Вновь получились два датасета размерностями (160, 8, 2), где 2 означает, что сохраняли индекс объекта и корреляцию. 

## Наше решение

Наше решение заключается в том, что мы объединяли пары самых скоррелированных объектов, используя созданные датасеты. Соответственно получали разделение на 160 кластеров.

# Task 2

## Наше решение

Мы решили, что если данные в приватной выборке берутся из датасета IHB, то разумнее всего будет обучаться на этом же датасете. Так как сэмплов из него всего 20, то сжимать размерность с помощью PCA можно только до 20, но и на такой размерности при таком размере датасета модель практически сразу переобучалась, поэтому мы регуляризировали ее с помощью уменьшения размерности данных до 12 и обучали на всем датасете.
