# SubTranslate

**Краткое описание проекта.**
Этот проект создан для автоматического перевода субтитров, встроенных в видео. Проект реализован на Python, для обработки видео используется ffmpeg, через консоль и нейросеть "Helsinki-NLP/opus-mt-en-ru" из huggingface для перевода.

## Установка
Сначала требуется установить [ffmpeg](https://github.com/FFmpeg/FFmpeg?tab=readme-ov-file) и удостоверится, что он добавлен в path. Затем можно устанавливать этот репозиторий.
```bash
git clone https://github.com/saigedoc/SubTranslate
pip install -r requirements.txt
```

## Как пользоваться

- В папку input переместить ролики, субтитры которых хочется перевести.
- Запустить код.
- Переместить ролики из папки output в удобное для вас место.

**Примечания**

Время работы программы зависит впервую очередь от длинны ролика (точнее от количества прописанных диалогов в субтитрах) и используемого устройства (cpu/gpu). На моём пк на gpu 1 видео 24 минуты, содержащее 709 фраз? перевелось за 1:44, когда на cpu - 5:04.
Чтобы использовать gpu нужно иметь библиотеку pytorch c поддержкой cuda и саму cuda. Проверить работает ли можно так:
```python
import torch
print(torch.cuda.is_available())
```