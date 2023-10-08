<h1>Прогноз оттока клиентов банка</h1>

<h2>Описание задачи</h2>
<p>В данном проекте вам предстоит выполнить задачу классификации клиентов банка: построить модель, прогнозирующую отток клиента из банка, и оценить точность предсказания по метрике ROC-AUC на тестовой выборке.</p>

<h2>Исходные данные</h2>
<p>Для анализа и построения модели вам предоставлены следующие данные:</p>
<ul>
<li><strong>Тренировочный датасет:</strong> содержит 300 000 записей о клиентах банка и значения целевых переменных для каждого из них. Описание факторов из датасета доступно в файле.</li>
</ul>

<h2>Целевая переменная</h2>
<p>Целевая переменная представлена в трёх столбцах:</p>
<ul>
<li><code>target1</code>: прекращение финансовой активности клиента.</li>
<li><code>target2</code>: закрытие РКО (расчетно-кассового обслуживания).</li>
<li><code>total_target</code>: максимальное значение из <code>target1</code> и <code>target2</code>, представляющее отток клиента из банка.</li>
</ul>

<h2>Задача</h2>
<p>Ваша задача состоит в следующем:</p>
<ol>
<li>Построить модель, которая предсказывает значение <code>total_target</code>.</li>
<li>С помощью построенной модели выполнить предсказание на тестовой выборке, которая содержит 100 000 записей.</li>
</ol>

<h2>Метрика</h2>
<p>Оценка качества модели будет проводиться с использованием метрики ROC-AUC.</p>
<h2>Инструкции</h2>
<p>Для выполнения задачи вам потребуются следующие шаги:</p>
<ol>
<li>Загрузите данные из файлов <code>train.csv</code> и <code>test.csv</code>.</li>
<li>Изучите описание факторов из файла <code>feature_description.xlsx</code>, чтобы понять, какие признаки доступны для анализа.</li>
<li>Проведите предварительный анализ данных, включая визуализацию и обработку пропущенных значений.</li>
<li>Выберите подходящие признаки для построения модели и выполните их подготовку.</li>
<li>Разделите тренировочные данные на обучающую и валидационную выборки.</li>
<li>Обучите модель машинного обучения для классификации, используя выбранные признаки.</li>
<li>Оцените качество модели на валидационной выборке с использованием метрики ROC-AUC.</li>
<li>Примените обученную модель для предсказания значений <code>total_target</code> на тестовой выборке из файла <code>test.csv</code>.</li>
<li>Подготовьте результаты предсказания для дальнейшей оценки.</li>
</ol>

<h2>Зависимости</h2>
<p>Для выполнения этой задачи могут понадобиться следующие библиотеки Python:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>scikit-learn</li>
<li>matplotlib</li>
<li>seaborn</li>
</ul>
