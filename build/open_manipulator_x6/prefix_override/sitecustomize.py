import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/gilbertsoco/Open-Manipulator-X-6/install/open_manipulator_x6'
