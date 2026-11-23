# Paper Mending Bench

Compare reflected and transmitted photographs of paper before a conservation repair. Import and calibrate the sheets, mark regions or transects, and save image measurements for each inspection.

Run `./start.sh` and open `http://127.0.0.1:19882/`. Open the demonstration sheet to try three explicitly synthetic photographs. The saved workspace is on the `paper-library` Compose volume. Use `./start.sh --down` to stop the service while retaining the volume.

The bench supports JPEG, PNG, and TIFF up to 20 MiB and 14 megapixels per image. Each inspection holds up to twelve photos and one hundred annotations. Enter the physical dimensions and mark four corresponding corners of each photograph. The calibrated raster is capped at 1300 pixels on its longer edge with at most four pixels per millimetre.

The readings cover sheet edges, backlit loss regions, tear cross-sections, trial patch coverage, fiber direction, ink intervals, foxing, paper color, landmark shifts, shared visibility, transmission differences, watermark contrast, and fold ligaments. Saved reports include the result overlay, markings, settings, exact original photographs and CSV.

The color, brightness and connectivity measurements are image estimates and depend on lighting, visibility and calibration. Compare results under controlled light. Coordinates refer to the calibrated plane.
