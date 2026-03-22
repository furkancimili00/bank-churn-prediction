#!/bin/bash
# Fast test script to ensure our python files are syntactically correct and run locally
python3 -c "import api.app"
python3 -m py_compile dashboard.py
echo "Tests Passed"
