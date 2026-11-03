CarVision 2.12 PHONE — полный выпуск с приёмом фото с телефона.

## Новая установка — телефон уже встроен

Скачайте `CarVision-2.12-Phone-Full.zip.001`, `.002`, `RESTORE_FULL.cmd`
и `RESTORE_FULL.ps1` в одну пустую папку. Запустите `RESTORE_FULL.cmd`,
дождитесь распаковки, затем откройте `CarVision/CarVision.exe`.
В заголовке будет **2.12 PHONE**. Старая версия и патчи не нужны.

В приложении нажмите **С телефона**, подключите устройства к одной Wi-Fi-сети
и откройте QR-ссылку адреса Wi-Fi компьютера. **Выбрать фото** открывает
галерею, **Снять камерой** — камеру телефона. После выбора и подтверждения
кадра анализ запускается автоматически. Результат виден на обоих устройствах.

Владелец подтвердил работу галереи на реальном телефоне и отображение
ответа на телефоне и компьютере. Камера открывается; отправку нового
снимка ещё предстоит проверить. Физическое USB-подключение не проверено.

## Обновление уже установленной 2.12 TEST

Скачайте `CarVision-2.12-Phone-Patch.zip`, распакуйте отдельно, закройте
CarVision и запустите `APPLY_PHONE_PATCH.cmd`. Выберите папку, в которой
лежит исходный `CarVision.exe` версии 2.12 TEST. После установки запустите
его снова: заголовок будет **2.12 PHONE**. Кнопка **С телефона** показывает
локальные адреса и QR-код. Веса модели не заменяются. Python не нужен.

- Gallery upload and native-camera snapshots automatically trigger analysis.
- Predictions and an annotated image appear on the computer and phone page.
- Shared Wi-Fi or a phone hotspot; Android USB tethering is also supported.
- Optional sequential browser-camera mode over Android USB/ADB localhost.
  ADB is not bundled. The Russian guide explains setup and browser limits.
- Bounded single-job processing, unknown preservation, upload validation,
  session links, temporary-image retention and explicit receiver shutdown.
- Hash-checked installation with backups; reapplying the same patch is safe.

Verified on this PC: 163 source tests; real Windows EXE with Python removed
from PATH; local HTTP upload and polling; result preview; identical direct/phone
brand-colour-model outputs for the same normalised development image;
the existing prediction, selected-region and export self-checks.
The browser file picker also completed an upload and showed real results.
Newly captured camera submissions, physical USB drivers and other recipient
networks still require a device test. This release does not claim better recognition accuracy.

Full Russian guide: `READ_FIRST_RU.txt` in the patch and `PHONE_INPUT_RU.txt`
in the installed app. The original `v2.12-test` release and `midterm` tag
are unchanged. This release includes both the self-contained full version and
the optional update for existing 2.12 TEST installations. Training crops and the
original local-photo archive remain available in the previous v2.12-test release.
