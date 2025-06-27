# Sprint Plan – Next Development Cycle

This sprint covers approximately two weeks of work aimed at preparing the beta release.

## Goals
- Close outstanding issues from the Beta-Ready Checklist.
- Improve cross-platform packaging instructions.
- Enhance unit test coverage for error conditions.

## Tasks
1. **Update Documentation**
   - Finalize PRD and Roadmap.
   - Write step-by-step setup guide in the README.
2. **GUI Polishing**
   - Add error messages for invalid dataset paths or target columns.
   - Implement file dialogs for saving/loading models.
3. **Testing**
   - Expand `tests/test_model_trainer_core.py` with additional edge cases.
   - Verify the GUI launches on Windows and Linux with sample data.
4. **Packaging**
   - Research PyInstaller configuration for building standalone binaries.
   - Document packaging steps in a new `PACKAGING.md` file.

## Deliverables
- Updated documentation and checklist
- Passing test suite (`pytest -q`)
- Draft packaging instructions

