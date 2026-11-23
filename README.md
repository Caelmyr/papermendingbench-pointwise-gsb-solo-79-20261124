# Paper Mending Bench

Compare reflected and transmitted photographs of paper before a conservation repair. Import and calibrate the sheets, mark regions or transects, and save image measurements for each inspection.

Run `./start.sh` and open `http://127.0.0.1:19882/`. Open the demonstration sheet to try three explicitly synthetic photographs. The saved workspace is on the `paper-library` Compose volume. Use `./start.sh --down` to stop the service while retaining the volume.

The bench supports JPEG, PNG, and TIFF up to 20 MiB and 14 megapixels per image. Each inspection holds up to twelve photos and one hundred annotations. Enter the physical dimensions and mark four corresponding corners of each photograph. The calibrated raster is capped at 1300 pixels on its longer edge with at most four pixels per millimetre.

The readings cover sheet edges, backlit loss regions, tear cross-sections, trial patch coverage, fiber direction, ink intervals, foxing, paper color, landmark shifts, shared visibility, transmission differences, watermark contrast, and fold ligaments. Saved reports include the result overlay, markings, settings, exact original photographs and CSV.

The sampled foxing reading uses a border band and a centre field marked as two disjoint strata, plus two alternative sets of sampling windows marked with the foxing sampling-window purpose. Each set is checked against the sampling-area cap you enter, counting the union of that set's windows once, and against the observed-area minimum for the band and for the centre. A set qualifies only if it meets the cap and both minima. When both qualify, one set is adopted before any stain is detected, by the set that gives the smaller stratum more observed sample; the greater total observed area breaks a tie, then Set A. Adjacent windows in one set form one union, so a stain on their shared edge is counted once. Only the adopted set is measured. The report gives the observed sampled area, the stained area and the stain percentage for the band and the centre separately, then a photograph-wide estimate weighted by each stratum's visible area in that photograph. Visible-but-unsampled, unreadable, and outside-the-photograph areas are reported as their own states and never as sampled and clear.

The color, brightness and connectivity measurements are image estimates and depend on lighting, visibility and calibration. Compare results under controlled light. Coordinates refer to the calibrated plane.
