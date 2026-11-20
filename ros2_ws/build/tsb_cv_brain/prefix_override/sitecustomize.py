import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/xandas/Desktop/tecnico/SA/Recrutamento/recrutamento2026/install/tsb_cv_brain'
