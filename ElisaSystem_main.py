import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap, QAction, QTextCursor, QColor, QTextCharFormat, QImage, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGraphicsView, QGraphicsScene, \
    QGraphicsPixmapItem, QTextBrowser, QFileDialog, QCheckBox, QRadioButton, QMessageBox, QSpinBox, QDoubleSpinBox, \
    QProgressBar
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Signal
from PySide6.QtCore import Qt

from inferThread import InferenceThread, BatchInferenceThread
from overlay_mask_on_image import overlay_mask_on_image, numpy_to_qpixmap
from postProcess import postprocess_mask
from prepareClassification import preprocess_for_classification


class MainWindow(QMainWindow):

    #定义信号
    model_changed = Signal(str,str) #模型改变

    def __init__(self):
        super().__init__()


        self.load_ui()
        self.setup_controls()
        self.setup_events()

        # 管理图片状态
        self.current_image_path = None #当前图片路径
        self.current_pixmap = None #当前图片QPixmap对象
        self.current_array = None #当前图片numpy数组
        self.current_patches_on_image =[]
        self.current_bboxes_on_image = []
        self.current_mask_on_image_idx = None

        # 批量图片状态
        self.image_folder_path = None #图片文件夹路径
        self.image_list = [] #图片列表
        self.current_image_index = 0 #图片索引
        self.save_path = None #保存路径
        self.BOOL_saveSeg = False #保存分割结果勾选框

        self.setWindowTitle("Elisa图像处理系统")
        self.setWindowIcon(QIcon(r"icon.png"))
        # 管理模型状态
        self.current_model = None #当前模型名称
        self.current_model_path = None #当前模型.pt路径

        # 分割参数控制
        self.use_gpu = False# 是否启用GPU推理
        self.stride = 192# 推理时patch移动步长
        self.confidence = 0.5# 置信度

        # 后处理参数控制
        self.circularity = 0.5# 圆度阈值
        self.min_area = 1500# 面积阈值
        self.current_prediction = None
        self.current_mask = None
        self.applyPostProcess = None

        # 可视化状态
        self.current_overlaid = None #当前可视化结果QPixmap对象


    #-----------------------------------初始化UI------------------------------------------------------
    def load_ui(self):
        # 打开 UI 文件
        ui_file = QFile(r"D:\QTproject\elisa\form.ui")
        ui_file.open(QFile.ReadOnly)
        # 加载 UI
        loader = QUiLoader()
        self.ui = loader.load(ui_file)  # self 作为父窗口
        ui_file.close()
        # 将 UI 设置为主窗口中心控件（可选，取决于 UI 文件布局）
        self.setCentralWidget(self.ui)

    #---------------------------------获取控件---------------------------------------------------------
    def setup_controls(self):
        self.btn_openSingleImage = self.ui.findChild(QPushButton, "pushButtonSingleImage")#打开单张图片按钮
        self.radioButton_useGPU = self.ui.findChild(QRadioButton, "radioButtonUsingGPU")#是否使用GPU推理按钮
        self.graphicsView_in = self.ui.findChild(QGraphicsView, "graphicsViewInput")#上传的图片显示区
        self.graphicsView_out = self.ui.findChild(QGraphicsView, "graphicsViewOutput")#输出图片显示区
        self.textBrowser = self.ui.findChild(QTextBrowser, "textBrowserLog")#日志显示区

        self.spinBox_stride = self.ui.findChild(QSpinBox, "spinBoxStride")#patch移动步长
        self.spinBox_stride.setValue(192)# 设置初始显示值

        self.spinBox_confidence = self.ui.findChild(QDoubleSpinBox, "doubleSpinBoxConfidence")#置信度阈值
        self.spinBox_confidence.setValue(0.5)# 设置初始显示值

        self.btn_segmentation = self.ui.findChild(QPushButton, "pushButtonSegmentation")#分割按钮
        self.btn_singlePostProcess = self.ui.findChild(QPushButton, "pushButtonSegPostProcess")# 后处理按钮
        self.checkBox_PostProcess = self.ui.findChild(QCheckBox, "checkBoxSegPostProcess")# 优化后的后处理按钮，可以通过勾选在是否后处理结果中切换

        self.progressBar_patch = self.ui.findChild(QProgressBar, "progressBarPatch")#进度条
        self.progressBar_patch.setValue(0)  # 设置初始值
        self.progressBar_batch = self.ui.findChild(QProgressBar, "progressBarBatch")#进度条
        self.progressBar_batch.setValue(0)  # 设置初始值

        self.btn_openMutilImageFolder = self.ui.findChild(QPushButton, "pushButtonMutiImage")#打开多张图片按钮
        self.btn_Savepath = self.ui.findChild(QPushButton, "pushButtonSavepath")#保存路径按钮

        self.checkBox_saveSeg = self.ui.findChild(QCheckBox, "checkBoxSaveSeg")#是否保存分割结果按钮
        self.checkBox_saveSeg.setEnabled(False) # 初始状态为不可用

        self.btn_startBatch = self.ui.findChild(QPushButton, "pushButtonStartBatch")#开始批量推理按钮

        self.spinBox_circularity = self.ui.findChild(QDoubleSpinBox, "doubleSpinBoxCircularity")#圆形度阈值
        self.spinBox_circularity.setValue(0.5)  # 设置初始显示值

        self.spinBox_minArea = self.ui.findChild(QSpinBox, "spinBoxArea")#面积阈值
        self.spinBox_minArea.setValue(1500)#设置初始显示值

        self.btn_ROI = self.ui.findChild(QPushButton, "pushButtonROI") #根据掩码区域裁切ROI


        self.btn_selectUnet = self.ui.findChild(QAction, "actionUnet")#选择UNet
        self.btn_selectUnetWeight = self.ui.findChild(QAction, "actionUnet_weight")#选择Unet权重文件
        self.btn_selectAttentionUnet = self.ui.findChild(QAction, "actionAttentionUnet")#选择AttentionUNet
        self.btn_selectAttentionUnetWeight = self.ui.findChild(QAction, "actionAttentionUnet_weight")#选择AttentionUnet权重文件
        self.btn_selectR2Unet = self.ui.findChild(QAction, "actionR2Unet")#选择R2Unet
        self.btn_selectR2UnetWeight = self.ui.findChild(QAction, "actionR2Unet_weight")#选择R2Unet权重文件
        self.btn_selectSwinUnet = self.ui.findChild(QAction, "actionSwinUnet")#选择SwinUnet
        self.btn_selectSwinUnetWeight = self.ui.findChild(QAction, "actionSwinUnet_weight")# 选择SwinUnet权重文件
        self.btn_selectFCN8S = self.ui.findChild(QAction, "actionFCN8S")#选择FCN8S
        self.btn_selectFCN8SWeight = self.ui.findChild(QAction, "actionFCN8S_weight")#选择FCN8S权重文件

        self.label_modelInfo = self.ui.findChild(QLabel, "labelModel")#模型信息显示区
        self.label_modelWeightInfo = self.ui.findChild(QLabel, "labelWeight")#模型权重信息显示区


    #----------------------------------事件绑定-----------------------------------------------------
    def setup_events(self):
        #点击打开图片按钮
        self.btn_openSingleImage.clicked.connect(self.open_image_dialog)
        # 勾选是否使用GPU推理
        self.radioButton_useGPU.toggled.connect(self.use_gpu_changed)
        # 步长设置显示
        self.spinBox_stride.valueChanged.connect(self.stride_changed)
        # 置信度设置显示
        self.spinBox_confidence.valueChanged.connect(self.confidence_changed)
        # 圆形度阈值设置显示
        self.spinBox_circularity.valueChanged.connect(self.circularity_changed)
        # 面积阈值设置显示
        self.spinBox_minArea.valueChanged.connect(self.area_changed)
        # 分割按钮
        self.btn_segmentation.clicked.connect(self.start_single_inference)
        # 打开多张图片文件夹按钮
        self.btn_openMutilImageFolder.clicked.connect(self.select_image_folder)
        # 保存路径按钮
        self.btn_Savepath.clicked.connect(self.select_save_path)
        # 勾选框更新状态
        self.checkBox_saveSeg.toggled.connect(self.update_checkbox_saveSeg_status)
        # 开始批量推理按钮
        self.btn_startBatch.clicked.connect(self.start_batch_inference)
        # 单张图片后处理
        self.btn_singlePostProcess.clicked.connect(self.start_single_post_process)
        self.checkBox_PostProcess.toggled.connect(self.update_checkbox_post_process_status)
        # 裁切ROI
        self.btn_ROI.clicked.connect(self.crop_ROI_base_on_mask)


        #选择Unet模型
        self.btn_selectUnet.triggered.connect(self.select_unet)
        #选择Unet权重文件
        self.btn_selectUnetWeight.triggered.connect(self.select_unet_weight)
        #选择AttentionUnet模型
        self.btn_selectAttentionUnet.triggered.connect(self.select_attention_unet)
        #选择AttentionUnet权重文件
        self.btn_selectAttentionUnetWeight.triggered.connect(self.select_attention_unet_weight)
        #选择R2Unet模型
        self.btn_selectR2Unet.triggered.connect(self.select_r2_unet)
        #选择R2Unet权重文件
        self.btn_selectR2UnetWeight.triggered.connect(self.select_r2_unet_weight)
        #选择SwinUnet模型
        self.btn_selectSwinUnet.triggered.connect(self.select_swin_unet)
        #选择SwinUnet权重文件
        self.btn_selectSwinUnetWeight.triggered.connect(self.select_swin_unet_weight)
        #选择FCN8S模型
        self.btn_selectFCN8S.triggered.connect(self.select_fcn8s)
        #选择FCN8S权重文件
        self.btn_selectFCN8SWeight.triggered.connect(self.select_fcn8s_weight)
        #选择不同模型时更新标签
        self.model_changed.connect(self.update_model_info)

    #---------------------------------功能函数-----------------------------------------------------
    # 中文路径读取
    def imread_unicode(self,file_path):
        """支持中文路径的读取"""
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 或 cv2.IMREAD_GRAYSCALE
        return img

    # 新对话框读取图片路径
    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            # 更新状态
            self.current_image_path = file_path
            self.current_pixmap = QPixmap(file_path)
            # 保存原始数组用于后续处理
            # self.current_array = cv2.imread(file_path)
            self.current_array = self.imread_unicode(file_path)
            self.add_log(f"打开图片：{file_path}")
            self.display_image_from_path(file_path)

    # 选择多张图片文件夹
    def select_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder_path:
            return
        self.image_folder_path = folder_path
        self.image_list = [os.path.join(folder_path, f)
                           for f in os.listdir(folder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        if not self.image_list:
            QMessageBox.warning(self, "警告", "未找到图片")
            return
        self.current_image_idx = -1 # 当前未显示图片
        self.add_log(f"共找到 {len(self.image_list)} 张图片")

    # 选择保存路径
    def select_save_path(self):
        self.save_path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if not self.save_path:
            return
        self.add_log(f"已选择保存路径：{self.save_path}")
        self.checkBox_saveSeg.setEnabled(True)

    # 检测gpu是否启用
    def use_gpu_changed(self,checked:bool):
        if checked:
            self.use_gpu = True
            self.add_log("开启GPU推理")
        else:
            self.use_gpu = False
            self.add_log("关闭GPU推理")


    # 步长调整
    def stride_changed(self,value:int):
        self.stride = value
        self.add_log(f"步长调整为：{value}")

    # 置信度调整
    def confidence_changed(self,value:float):
        self.confidence = value
        self.add_log(f"置信度调整为：{value:.2f}")

    # 圆形度阈值调整
    def circularity_changed(self,value:float):
        self.circularity = value
        self.add_log(f"后处理圆度阈值调整为：{value:.2f}")

    # 面积阈值调整
    def area_changed(self,value:int):
        self.min_area = value
        self.add_log(f"后处理面积阈值调整为：{value}")

    # graphicsView显示图片
    def display_image_from_path(self, image_path):
        """在 QGraphicsView 显示图片，可复用"""
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView_in.setScene(scene)
        self.graphicsView_in.fitInView(item,Qt.KeepAspectRatio)  # 自动缩放填充

    # 源是pixmap，显示到graphicsview
    def display_image_in_graphicsview(self, pixmap: QPixmap):
        scene = QGraphicsScene()  # 创建场景
        scene.addPixmap(pixmap)  # 添加图片
        self.graphicsView_out.setScene(scene)
        self.graphicsView_out.fitInView(scene.itemsBoundingRect(),Qt.KeepAspectRatio)  # 自动缩放适应

    # 日志记录
    def add_log(self, message: str):
        """在 textBrowser 中添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        # 设置时间戳的颜色为绿色
        format = QTextCharFormat()
        format.setForeground(QColor("green"))
        cursor.setCharFormat(format)
        cursor.insertText(f"[{timestamp}] ")
        # 重置文本格式，以便后续的消息以默认格式显示
        format.setForeground(QColor("black"))
        cursor.setCharFormat(format)
        cursor.insertText(f"{message}\n")
        # 自动滚动到末尾
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    # 手动后处理
    def start_single_post_process(self):
        if self.current_prediction is None:
            QMessageBox.warning(self, "警告", "没有可以后处理的分割结果")
            return
        processed_mask,num_objects = postprocess_mask(self.current_prediction,
                                                      max_objects=96,
                                                      min_area=self.min_area,
                                                      min_circularity=self.circularity)
        # 将数量写入日志
        self.add_log(f"后处理完成，共找到 {num_objects} 个有效孔")
        # 将 numpy 转 QPixmap 显示到 graphicsView_out
        self.current_mask = processed_mask#更新当前掩码
        height, width = processed_mask.shape
        bytes_per_line = width
        q_image = QImage(processed_mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_out.setScene(scene)
        self.graphicsView_out.fitInView(scene.itemsBoundingRect(),Qt.KeepAspectRatio)  # 自动缩放适应
        # 原图重绘
        self.on_inference_finished(self.current_mask)

    #选择Unet模型
    def select_unet(self):
        # 设置当前模型路径/类型
        self.current_model = "Unet"  # 或者 ModelType.UNET.name
        #发出信号，更新标签
        self.model_changed.emit("Unet",'')
        #日志记录
        self.add_log(f"选择模型：{self.current_model}")
    #选择Unet权重文件
    def select_unet_weight(self):
        # 设置当前模型权重路径
        file_path = 'weight/UNet_16_N_ElisaTemplateDataset256_150_seed42_best.pth'
        if file_path:
            self.current_model_path = file_path
            self.model_changed.emit('',file_path)
            self.add_log(f"选择模型权重文件：{file_path}")

    #选择AttentionUnet模型
    def select_attention_unet(self):
        # 设置当前模型路径/类型
        self.current_model = "AttentionUnet"  # 或者 ModelType.ATTENTION_UNET.name
        # 发出信号，更新标签
        self.model_changed.emit("AttentionUnet",'')
        # 日志记录
        self.add_log(f"选择模型：{self.current_model}")
    #选择AttentionUnet权重文件
    def select_attention_unet_weight(self):
        # 设置当前模型权重路径
        file_path = 'weight/Attention_UNet_16_N_ElisaTemplateDataset256_150_seed42_best.pth'
        if file_path:
            self.current_model_path = file_path
            self.model_changed.emit('',file_path)
            self.add_log(f"选择模型权重文件：{file_path}")

    # 选择模型R2Unet
    def select_r2_unet(self):
        # 设置当前模型路径/类型
        self.current_model = "R2Unet"  # 或者 ModelType.R2_UNET.name
        # 发出信号，更新标签
        self.model_changed.emit("R2Unet",'')
        # 日志记录
        self.add_log(f"选择模型：{self.current_model}")
    #选择R2Unet权重文件
    def select_r2_unet_weight(self):
        # 设置当前模型权重路径
        file_path = 'weight/r2unet_16_N_ElisaTemplateDataset256_150_seed42_best.pth'
        if file_path:
            self.current_model_path = file_path
            self.model_changed.emit('',file_path)
            self.add_log(f"选择模型权重文件：{file_path}")

    # 选择SwinUnet模型
    def select_swin_unet(self):
        # 设置当前模型路径/类型
        self.current_model = "SwinUnet"  # 或者 ModelType.SWIN_UNET.name
        # 发出信号，更新标签
        self.model_changed.emit("SwinUnet",'')
        # 日志记录
        self.add_log(f"选择模型：{self.current_model}")
    #选择SwinUnet权重文件
    def select_swin_unet_weight(self):
        # 设置当前模型权重路径
        file_path = 'weight/SwinUNet_16_N_ElisaTemplateDataset256_150_seed42_best.pth'
        if file_path:
            self.current_model_path = file_path
            self.model_changed.emit('',file_path)
            self.add_log(f"选择模型权重文件：{file_path}")

    # 选择FCN8S模型
    def select_fcn8s(self):
        # 设置当前模型路径/类型
        self.current_model = "FCN8S"  # 或者 ModelType.FCN8S.name
        # 发出信号，更新标签
        self.model_changed.emit("FCN8S",'')
        # 日志记录
        self.add_log(f"选择模型：{self.current_model}")
    #选择FCN8S权重文件
    def select_fcn8s_weight(self):
        # 设置当前模型权重路径
        file_path = 'weight/fcn8s_16_N_ElisaTemplateDataset256_150_seed42_best.pth'
        if file_path:
            self.current_model_path = file_path
            self.model_changed.emit('',file_path)
            self.add_log(f"选择模型权重文件：{file_path}")

    # 更新模型信息
    def update_model_info(self,model_name:str,model_weight:str):
        """更新模型信息"""
        if self.current_model:
            self.label_modelInfo.setText(f"当前选择模型：{self.current_model}")
        else:
            self.label_modelInfo.setText("当前选择模型：未选择")
        if self.current_model_path:
            self.label_modelWeightInfo.setText(f"模型权重路径：{self.current_model_path}")
        else:
            self.label_modelWeightInfo.setText("模型权重路径：未选择")

    # 更新进度条
    def update_progress_bar(self,patch_count,total_patch):
        self.progressBar_patch.setMaximum(total_patch)
        self.progressBar_patch.setValue(patch_count)
    def update_progress_bar_batch(self,patch_count,total_patch):
        self.progressBar_batch.setMaximum(total_patch)
        self.progressBar_batch.setValue(patch_count)

    # 推理线程
    def start_single_inference(self):
        if self.current_image_path is None:
            QMessageBox.warning(self, "警告", "请上传选择图片")
            return
        if self.current_model is None:
            QMessageBox.warning(self, "警告", "请选择模型")
            return
        if self.current_model_path is None:
            QMessageBox.warning(self, "警告", "请选择模型权重文件")
            return
        # 准备清空输出显示区
        if not self.graphicsView_out.scene():
            self.graphicsView_out.setScene(QGraphicsScene(self))
        # 每次推理前清空
        self.graphicsView_out.scene().clear()
        #每次推理开始时清空进度条
        self.progressBar_patch.setValue(0)
        # 创建线程
        self.infer_thread = InferenceThread(
            self.current_model,self.current_model_path, self.current_image_path, self.stride,self.confidence,self.use_gpu
        )
        # cuda环境检查
        self.infer_thread.cuda_available.connect(self.handle_device_status)
        # 连接信号，将结果更新到 UI
        self.infer_thread.result_ready.connect(self.handle_inference_result)
        # 进度条更新
        self.infer_thread.progress_changed.connect(self.update_progress_bar)
        # 启动线程
        self.infer_thread.start()
        self.add_log("推理已开始...")


    # 统一调整勾选框状态
    def update_checkbox_saveSeg_status(self,checked):
        # 记录勾选状态
        # 保存批量分割结果勾选框
        if checked:
            self.BOOL_saveSeg = True
            self.add_log("保存分割结果")
        else:
            self.BOOL_saveSeg = False
            self.add_log("不保存分割结果")
        # 单张图片后处理勾选框

    def update_checkbox_post_process_status(self, checked:bool):
        if checked and self.current_prediction is None:
            QMessageBox.information(self, "提示", "已经开启后处理")
            return

        if checked:
            # 进行后处理
            processed_mask, num_objects = postprocess_mask(
                self.current_prediction,
                max_objects=96,
                min_area=self.min_area,
                min_circularity=self.circularity
            )
            self.current_mask = processed_mask
            self.add_log(f"开启后处理，共找到 {num_objects} 个有效孔")
        else:
            # 恢复原始预测
            self.current_mask = self.current_prediction
            self.add_log("已取消后处理，显示原始预测结果")

        # 无论哪种情况，只要存在预测结果，都更新显示
        if self.current_mask is not None:
            height, width = self.current_mask.shape
            q_image = QImage(
                self.current_mask.data,
                width,
                height,
                width,
                QImage.Format_Grayscale8
            )
            pixmap = QPixmap.fromImage(q_image)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView_out.setScene(scene)
            self.graphicsView_out.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # 如果你需要叠加显示区域（彩色连通域），可以在这里调用
            self.on_inference_finished(self.current_mask)




    # batch图片推理线程
    def start_batch_inference(self):
        if self.image_folder_path is None:
            QMessageBox.warning(self, "警告", "请选择图片文件夹")
            return
        if len(self.image_list)==0:
            QMessageBox.warning(self, "警告", "文件夹内没有图片")
            return
        if self.current_model is None:
            QMessageBox.warning(self, "警告", "请选择模型")
            return
        if self.current_model_path is None:
            QMessageBox.warning(self, "警告", "请选择模型权重文件")
            return
        self.applyPostProcess = self.checkBox_PostProcess.isChecked()
        # 开始推理，禁用按钮
        self.disable_buttons()
        # 每次批次推理，清空输入区索引
        self.current_image_index = 0
        self.batch_infer_thread = BatchInferenceThread(
            self.current_model,
            self.current_model_path,
            self.image_list,
            self.stride,
            self.confidence,
            self.use_gpu,
            self.save_path,
            self.BOOL_saveSeg,
            self.applyPostProcess,
            max_objects=96,
            min_area=self.min_area,
            min_circularity=self.circularity

        )
        # 绑定信号
        self.batch_infer_thread.single_progress.connect(self.update_progress_bar)#单个进度条更新
        self.batch_infer_thread.batch_progress.connect(self.update_progress_bar_batch)# 整个batch进度
        self.batch_infer_thread.finished_single.connect(self.handle_single_result)# 单个推理结果处理，包含上传区的下张图片显示
        self.batch_infer_thread.finished_batch.connect(lambda: self.textBrowser.append("批量推理完成"))# 整个batch推理完成
        self.batch_infer_thread.finished_batch.connect(self.enable_buttons)# 启用按钮
        self.batch_infer_thread.cuda_available.connect(self.handle_device_status)# cuda环境检查

        # 启动线程
        self.batch_infer_thread.start()
        self.add_log("批量推理已开始...")

    # 禁用按钮
    def disable_buttons(self):
        self.checkBox_PostProcess.setEnabled(False)
        self.btn_Savepath.setEnabled(False)
        self.btn_openMutilImageFolder.setEnabled(False)
        self.spinBox_stride.setEnabled(False)
        self.spinBox_confidence.setEnabled(False)
        self.checkBox_saveSeg.setEnabled(False)
        self.spinBox_minArea.setEnabled(False)
        self.spinBox_circularity.setEnabled(False)
        self.radioButton_useGPU.setEnabled(False)
        self.btn_openSingleImage.setEnabled(False)
        self.btn_segmentation.setEnabled(False)
        self.btn_singlePostProcess.setEnabled(False)
        self.btn_startBatch.setEnabled(False)

    # 启用按钮
    def enable_buttons(self):
        self.checkBox_PostProcess.setEnabled(True)
        self.btn_Savepath.setEnabled(True)
        self.btn_openMutilImageFolder.setEnabled(True)
        self.spinBox_stride.setEnabled(True)
        self.spinBox_confidence.setEnabled(True)
        self.checkBox_saveSeg.setEnabled(True)
        self.spinBox_minArea.setEnabled(True)
        self.spinBox_circularity.setEnabled(True)
        self.radioButton_useGPU.setEnabled(True)
        self.btn_openSingleImage.setEnabled(True)
        self.btn_segmentation.setEnabled(True)
        self.btn_singlePostProcess.setEnabled(True)
        self.btn_startBatch.setEnabled(True)


    # 转换推理结果为QPixmap
    def convert_output_to_pixmap(self,array: np.ndarray) -> QPixmap:
        """
        将 NumPy 数组转为 QPixmap，方便在 QGraphicsView 或 QLabel 显示
        array: np.ndarray
            - 2D 灰度图像或 3D 彩色图像 (H, W, 3)
            - 数据类型最好是 uint8，值域 0~255
        """
        # 如果是浮点数，先归一化到 0~255
        if np.issubdtype(array.dtype, np.floating):
            array = (array * 255).clip(0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)

        # 2D 灰度图转换为 3D RGB
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)

        # 转 PIL Image
        img = Image.fromarray(array)

        # 转 QPixmap
        qpixmap = QPixmap.fromImage(ImageQt(img))
        return qpixmap

    # 处理推理结果
    def handle_inference_result(self,output,infer_time):
        # output 可以是 mask、概率图等
        self.add_log(f"推理完成,用时{infer_time:.2f}秒")
        # 保存到当前推理结果
        self.current_prediction = output
        # 判断后处理开关
        if self.checkBox_PostProcess.isChecked():
            processed_mask, num_objects = postprocess_mask(
                self.current_prediction,
                max_objects=96,
                min_area=self.min_area,
                min_circularity=self.circularity
            )
            self.current_mask = processed_mask
            self.add_log(f"已开启后处理，共找到 {num_objects} 个有效孔")
        else:
            self.current_mask = self.current_prediction
        # 根据最终的 current_mask 更新显示
        pixmap = self.convert_output_to_pixmap(self.current_mask)
        self.display_image_in_graphicsview(pixmap)
        # 显示叠加掩码
        self.on_inference_finished(self.current_mask)

    # 多个图片单张推理结果处理
    def handle_single_result(self,output,image_path,infer_time):
        # 保存当前推理结果
        self.current_prediction = output
        # 判断后处理是否开启
        if self.checkBox_PostProcess.isChecked():
            processed_mask, num_objects = postprocess_mask(
                self.current_prediction,
                max_objects=96,
                min_area=self.min_area,
                min_circularity=self.circularity
            )
            self.current_mask = processed_mask
        else:
            self.current_mask = self.current_prediction
        try:
            self.current_image_path = image_path
            self.current_array = self.imread_unicode(image_path)  # 保证 current_array 有当前大图数据
        except Exception as e:
            self.add_log(f"警告：无法读取 {image_path} 的原图用于叠加：{e}")
            self.current_array = None
        # 显示推理结果
        pixmap = self.convert_output_to_pixmap(self.current_mask)
        self.display_image_in_graphicsview(pixmap)
        self.add_log(f"{image_path}推理完成,用时{infer_time:.2f}秒")

        # 上传区自动显示下一张图片

        if self.current_image_index < len(self.image_list):
            self.display_image_from_path(self.image_list[self.current_image_index])
            self.current_image_index += 1
            if self.current_array is not None and self.current_mask is not None:
                self.on_inference_finished(self.current_mask)


    # 处理cuda环境检查结果
    def handle_device_status(self,available:bool):
        if self.radioButton_useGPU.isChecked() and not available:
            self.add_log("注意：CUDA不可用，将使用CPU推理")
        # else:
        #     self.textBrowser.append(f"使用{'GPU' if available else 'CPU'}进行推理")

    # 将掩码标注在原图上
    def on_inference_finished(self, mask: np.ndarray):
        # 叠加掩码
        overlaid, num_regions = overlay_mask_on_image(self.current_array, mask)
        # 更新日志
        # self.add_log(f"已经在原图上绘制 {num_regions} 个目标")

        # 显示在 graphicsView_out
        pixmap = numpy_to_qpixmap(overlaid)
        self.graphicsView_in.scene().clear()
        self.graphicsView_in.scene().addPixmap(pixmap)
        self.graphicsView_in.fitInView(self.graphicsView_in.scene().itemsBoundingRect(),Qt.KeepAspectRatio)

        # 保存处理后的 mask，方便后续点击后处理按钮
        # self.current_prediction = mask
        self.current_overlaid = overlaid

    # 裁切ROI区域
    def crop_ROI_base_on_mask(self):
        #判断当前掩码是否为空
        if self.current_mask is None:
            QMessageBox.warning(self, "警告", "当前不存在掩码文件，请先进行分割")
            return
        self.current_patches_on_image,self.current_bboxes_on_image = preprocess_for_classification(self.current_mask,self.current_array,patch_size=224)
        self.add_log(f"已裁切 {len(self.current_patches_on_image)} 个ROI区域")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
