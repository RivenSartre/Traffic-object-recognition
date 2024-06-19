from UI_core.Log_UI import UI_Log

from UI_core.Sign_UI import UI_Sign
from UI_core.Win_UI import UI_Win
from PyQt5.QtWidgets import QApplication, QLineEdit, QDialog

import sys

class Main_Pross():
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.UI_Log = UI_Log()
        self.UI_Sign = UI_Sign()
        self.UI_Win = UI_Win()
        self.UI_Log.ui.PassWordShow_Button.pressed.connect(self.UI_Log.ShowPassWord)
        self.UI_Log.ui.PassWordShow_Button.released.connect(self.UI_Log.HidePassWord)
        self.UI_Log.ui.Log_Button.clicked.connect(lambda: self.UI_Log.Log_Function(self.UI_Win))
        self.UI_Log.ui.Cancel_Button.clicked.connect(self.UI_Log.close)
        self.UI_Log.ui.Sign.clicked.connect(self.UI_Sign.show)
        self.UI_Sign.ui.PassWordShow_Button.pressed.connect(self.UI_Sign.ShowEdit1PassWord)
        self.UI_Sign.ui.PassWordShow_Button_4.pressed.connect(self.UI_Sign.ShowEdit2PassWord)
        self.UI_Sign.ui.PassWordShow_Button.released.connect(self.UI_Sign.HideEdit1PassWord)
        self.UI_Sign.ui.PassWordShow_Button_4.released.connect(self.UI_Sign.HideEdit2PassWord)
        self.UI_Sign.ui.Log_Button.clicked.connect(self.UI_Sign.Sign_Function)
        self.UI_Sign.ui.Cancel_Button.clicked.connect(self.UI_Sign.close)
        self.UI_Win.ui.comboBox.currentIndexChanged.connect(self.UI_Win.load_setting)
        self.UI_Win.ui.iouSpinBox.valueChanged.connect(self.UI_Win.load_setting)
        self.UI_Win.ui.confSpinBox.valueChanged.connect(self.UI_Win.load_setting)
        self.UI_Win.ui.iouSlider.valueChanged.connect(self.UI_Win.SliderChange)
        self.UI_Win.ui.confSlider.valueChanged.connect(self.UI_Win.SliderChange)
        self.UI_Win.ui.Hide_conf_CheckBox.stateChanged.connect(self.UI_Win.load_setting)
        self.UI_Win.ui.Hide_label_CheckBox.stateChanged.connect(self.UI_Win.load_setting)
        self.UI_Win.ui.runButton.clicked.connect(self.UI_Win.run)
        self.UI_Win.ui.stopButton.clicked.connect(self.UI_Win.stop)
        self.UI_Win.det.send_raw.connect(lambda x: self.UI_Win.show_image(x, self.UI_Win.ui.raw_video))
        self.UI_Win.det.send_img.connect(lambda x: self.UI_Win.show_image(x, self.UI_Win.ui.out_video))
        self.UI_Win.det.send_statistic.connect(self.UI_Win.show_statistic)
        self.UI_Win.det.send_msg.connect(lambda x: self.UI_Win.show_msg(x))
        self.UI_Win.det.send_percent.connect(lambda x: self.UI_Win.ui.progressBar.setValue(x))
        self.UI_Win.det.send_fps.connect(lambda x: self.UI_Win.ui.fps_label.setText(x))
        self.UI_Win.ui.minButton.clicked.connect(self.UI_Win.showMinimized)
        self.UI_Win.ui.maxButton.clicked.connect(self.UI_Win.max_or_restore)
        self.UI_Win.ui.maxButton.animateClick(10)
        self.UI_Win.ui.closeButton.clicked.connect(self.UI_Win.close)
        self.UI_Win.ui.fileButton.clicked.connect(self.UI_Win.open_file)
        self.UI_Win.ui.cameraButton.clicked.connect(self.UI_Win.CamreaMode)
        self.UI_Win.det.start()
        # self.UI_Win.close()
        # self.UI_Log.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    main_pro = Main_Pross()
