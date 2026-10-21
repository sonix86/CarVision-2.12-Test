import json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1];history=root/'docs/history-v20';history.mkdir(parents=True,exist_ok=True)
for name in ['README.md','VALIDATION.md']:
 if not (history/name).exists():shutil.copy2(root/name,history/name)
v=json.loads((root/'models/release_v21_validation.json').read_text(encoding='utf-8'))
(root/'README.md').write_text('''# CarVision2.1

София Рахманкулова. Распакуйте TashkentCarVision-v2.1-Portable.zip целиком и запустите CarVision/CarVision.exe. Переносится вся папка. Анализ локальный, без API, ключей и платежей. Для анализа Python не нужен.

Исправления: отключённый модуль эмблем больше не выполняется и не вызывает KeyError brand. Пустые/исключённые маски не создают неполных записей. При противоречии сетей марки и модели итоговая марка становится unknown, а заголовок показывает первый вариант модели с вопросительным знаком. Справа объясняется расхождение. Проценты разных сетей не являются сопоставимыми вероятностями истинности. Это честное воздержание, а не доказательство правильности первого варианта.

Учебное обучение сохраняется в models/own_training_candidate и не заменяет рабочие веса автоматически. «Проверка качества» использует завершённого кандидата и исходный набор; до обучения выводится объяснение. Разметка собственных10фото в data/own_oct21 исправлена под общий формат source/group. Все кадры одной поездки остаются одной группой: они не дают независимый train/validation сами по себе.

Нейронные веса сохранены от2.0. Рост точности китайских моделей в2.1 не заявляется. Доступны32семейства, а не все модели рынка. Предыдущий эксперимент на собственных фото и эмблемах отклонён; активные веса пока преимущественно/полностью из публичного обучения. Цвета при отражениях и близкие кузова остаются проблемой.

Документ docs/CarVision_Report_v2.1_DRAFT.pdf:6страниц,7разделов, источники, числа, схема, обучение, матрица ошибок, образцы собственных фото, AI disclosure и инструкция. Это актуальный проект отчёта: GitHub URL ещё не предоставлен, живой телефонный тест и достаточная база собственного обучения не закрыты. Исторический Prototype Report не является актуальным отчётом.

Для задания копируйте только brand colour. Модель с ? — предположение; unknown получает ноль за марку, но сохраняет возможные баллы за цвет. Лишние предсказания штрафуются по заданию. Не переписывайте правильные ответы вручную после просмотра эталона.

См. REQUIREMENTS_AUDIT.md для оставшихся условий, VALIDATION.md для реальных измерений. samples содержит отдельные JPEG и их источники. Оригиналы пользователя не получили публичную лицензию; публичная отправка не выполнена.
''',encoding='utf-8')
(root/'VALIDATION.md').write_text(f'''# Проверка2.1

78 автоматических тестов: регрессии отключённых эмблем, пустых масок, конфликтов классификаторов и кнопок обучения/оценки. Полный offline smoke обучения: эпоха классификатора, checkpoint, confusion/evaluation, эпоха резервного детектора. Синтетическая проверка подтверждает работу кода, не качество распознавания.

17 реальных исходных/диагностических фото обработаны без исключений (outputs/user-regressions-v21). Эти известные примеры не являются независимой оценкой.

На161 исходной validation-вырезке марка верна в{v['brand_accuracy']:.1%}; unknown={v['unknown']}, конфликтов сетей={v['conflicts']}. Матрица в models/release_v21_validation.json. Включены только марки этой выборки; это не scene AP, не20свежих фото, не гарантия100% по китайским машинам. Правило воздержания может снижать число верных ответов одновременно с устранением противоречивой уверенности.

Веса2.0 не изменены. Их416test-кропов и историческое сравнение сохранены в models/body_expert_evaluation.json; не выдавать модельную точность за точность марки. Отдельные тесты цвета также не равны end-to-end точности сцены. Нейронных параметров и отдельных компонентов: models/release_v21_parameters.json.

Финальная упаковка проверяет14замороженных модулей, исходный offline-конвейер и ROI/экспорт, фактический EXE из отдельной папки без Python в PATH,7SHA весов и CRC/SHA каждого файла архива. Протоколы: outputs/frozen-v21-check.json, release-runtime-v21/checks.json, portable-v21-check.json, archive-v21-check.log. Другой физический ПК и полноценный телефонный цикл ещё не проверены.
''',encoding='utf-8')
for name in ['REQUIREMENTS_AUDIT.md','CHECKLIST.md']:
 p=root/name;s=p.read_text(encoding='utf-8-sig');s=s.replace('21 октября 2026','22 октября 2026').replace('Исторический отчёт сохранён. Новый окончательный отчёт и GitHub URL ещё не готовы.','Подготовлен актуальный6-страничный DRAFT с7разделами, матрицей, примерами, параметрами и AI disclosure. GitHub URL не предоставлен; незавершённые академические пункты явно отмечены.');p.write_text(s,encoding='utf-8')
p=root/'TRANSFER.txt';p.write_text(p.read_text(encoding='utf-8-sig').replace('2.0','2.1'),encoding='utf-8')
p=root/'models/model_card.json';card=json.loads(p.read_text(encoding='utf-8'));card['release']='2.1';card['v21_changes']=dict(neural_weights_changed=False,conflict_abstention=True,disabled_badge_crash_fixed=True,training_output='models/own_training_candidate',brand_validation_accuracy=v['brand_accuracy'],validation_crops=161,validation_report='release_v21_validation.json');p.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
print('Current2.1 documentation and model-card written; historical2.0 retained')
