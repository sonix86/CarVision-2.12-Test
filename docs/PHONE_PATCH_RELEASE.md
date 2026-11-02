Phone input patch for the original CarVision 2.12 TEST Windows application.

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
Physical Android/iPhone cameras, USB drivers and the recipient's network
still require a device test. This patch does not claim better recognition accuracy.

Full Russian guide: `READ_FIRST_RU.txt` in the patch and `PHONE_INPUT_RU.txt`
in the installed app. The original `v2.12-test` release and `midterm` tag
are unchanged. This optional patch is published separately as a prerelease.
