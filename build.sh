#!/bin/bash
set -e

echo "📥 Скачиваем модель Vosk для русского языка..."
wget -q https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip

echo "📦 Распаковываем модель..."
unzip -q vosk-model-small-ru-0.22.zip

echo "🧹 Удаляем архив..."
rm vosk-model-small-ru-0.22.zip

echo "✅ Модель готова!"
