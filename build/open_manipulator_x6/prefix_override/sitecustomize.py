import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ri-one/open-manipulator-x-6/install/open_manipulator_x6'
