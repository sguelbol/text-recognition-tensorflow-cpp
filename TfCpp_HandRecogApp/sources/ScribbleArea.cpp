#include <QtWidgets>
#if defined(QT_PRINTSUPPORT_LIB)
#include <QtPrintSupport/qtprintsupportglobal.h>
#if QT_CONFIG(printdialog)
#include <QPrinter>
#include <QPrintDialog>
#include <QColor>

#endif
#endif

#include "../headers/ScribbleArea.h"
#include <opencv2/opencv.hpp>

#define singals tf_signals
#include "tensorflow/core/framework/tensor.h"
#include "../headers/Helper.h"
#undef signals


ScribbleArea::ScribbleArea(QWidget *parent) : QWidget(parent)
{
    setAttribute(Qt::WA_StaticContents);
    setFocusPolicy(Qt::StrongFocus);
    modified = false;
    scribbling = false;
    myPenWidth = 4;
    myPenColor = Qt::black;
    handwritingLayer = QImage(this->size(), QImage::Format_ARGB32);
    textLayer = QImage(this->size(), QImage::Format_ARGB32);
    handwritingLayer.fill(Qt::white);
    textLayer.fill(Qt::transparent);
}


void ScribbleArea::setModel(const std::shared_ptr<Model> model) {
    this->model = std::move(model);
}

bool ScribbleArea::openImage(const QString &fileName) {
    QImage loadedImage;
    if (!loadedImage.load(fileName)) {
        return false;
    }
    QSize newSize = loadedImage.size().expandedTo(size());
    resizeImage(&loadedImage, newSize, Qt::white);
    textLayer = loadedImage;
    handwritingLayer.fill(Qt::white);
    convertImageFormat(textLayer);
    makeWhitePixelsTransparent(textLayer);
    modified = false;
    update();
    return true;
}


bool ScribbleArea::saveImage(const QString &fileName, const char *fileFormat) {
    QImage visibleImage = textLayer;
    resizeImage(&visibleImage, QSize(width(), height()), Qt::white);
    makeTransparentPixelsWhite(&visibleImage);
    if (visibleImage.save(fileName, fileFormat)) {
        modified = false;
        return true;
    }
    return false;
}

void ScribbleArea::setPenColor(const QColor &newColor) {
    myPenColor = newColor;
}

void ScribbleArea::setPenWidth(int newWidth) {
    myPenWidth = newWidth;
}

void ScribbleArea::clearImage() {
    handwritingLayer.fill(Qt::white);
    textLayer.fill(Qt::transparent);
    modified = true;
    update();
}

void ScribbleArea::mousePressEvent(QMouseEvent *event) {
    if(event->button() == Qt::LeftButton) {
        lastPoint = event->pos();
        if (!scribbling) {
            x_min = 5000;
            x_max = -1;
            y_min = 5000;
            y_max = -1;
            scribbling = true;
        }

    }
}

void ScribbleArea::mouseMoveEvent(QMouseEvent *event) {
    if((event->buttons() & Qt::LeftButton) && scribbling) {
        drawLineTo(event->pos());
        if (event->pos().x() < x_min) {
            x_min = event->pos().x();
        }

        if (event->pos().x() > x_max) {
            x_max = event->pos().x();
        }

        if (event->pos().y() < y_min) {
            y_min = event->pos().y();
        }
        if (event->pos().y() > y_max) {
            y_max = event->pos().y();
        }
    }
}


void ScribbleArea::mouseReleaseEvent(QMouseEvent *event) {
    if(event->button() == Qt::LeftButton && scribbling) {
        if (!(event->modifiers() & Qt::ShiftModifier)) {
            drawLineTo(event->pos());
            scribbling = false;
            QImage extractedCharacter = extractWrittenCharacter();
            if (!extractedCharacter.isNull()) {
                cv::Mat transformed = transformTo28(&extractedCharacter);
                tensorflow::Tensor handwrittenChar = createTensorVector(transformed);
                int predictedCharacter = predictNumber(handwrittenChar);
                extractedCharacter.save(QString("/home/sguelbol/CodeContext/text_recognition_eink/noteApps/freeWritingApp/cmake-build-debug/untitled.png"), "PNG");
                auto [x, y, width, height] = calculateDimensionsForExtraction();
                drawCharOnTextLayer(x, y, width, height, predictedCharacter);
            }
        }
    }
}

void ScribbleArea::keyReleaseEvent(QKeyEvent *event) {
    if ((event->key() == Qt::Key_Shift) && !(QApplication::mouseButtons() & Qt::LeftButton)) {
        scribbling = false;
        QImage extractedCharacter = extractWrittenCharacter();
        if (!extractedCharacter.isNull()) {
            cv::Mat transformed = transformTo28(&extractedCharacter);
            tensorflow::Tensor handwrittenChar = createTensorVector(transformed);
            int predictedCharacter = predictNumber(handwrittenChar);
            extractedCharacter.save(QString("/home/sguelbol/CodeContext/text_recognition_eink/noteApps/freeWritingApp/cmake-build-debug/untitled.png"), "PNG");
            auto [x, y, width, height] = calculateDimensionsForExtraction();
            drawCharOnTextLayer(x, y, width, height, predictedCharacter);
        }
    }
}

QImage ScribbleArea::extractWrittenCharacter() {
    auto [x, y, width, height] = calculateDimensionsForExtraction();
    QImage letter = handwritingLayer.copy(x, y, width, height);
    handwritingLayer.fill(Qt::white);
    update();
    return letter;
}

cv::Mat ScribbleArea::transformTo28(QImage *letter) {
    cv::Mat input = QImageToCvMat(*letter);
    cv::Mat src = cv::Mat(28, 28, CV_8UC3);
    cvtColor(input, src, cv::COLOR_BGR2GRAY);
    cv::Mat resizedImage = cv::Mat(28, 28, CV_8UC3);
    cv::resize(src, resizedImage, cv::Size(28, 28), 0,0,cv::INTER_AREA);
    imshow("resized to 28x28", resizedImage);
    return resizedImage;
}

tensorflow::Tensor ScribbleArea::createTensorVector(cv::Mat& resizedImage) {
    tensorflow::Tensor tensorVector(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 784});
    auto tmap = tensorVector.tensor<float, 2>();
    for (int i = 0; i < resizedImage.rows; ++i) {
        for (int j = 0; j < resizedImage.cols; ++j) {
            tmap(0, 28*i+j) = static_cast<float>(resizedImage.at<uchar>(i, j));
        }
    }
    return tensorVector;
}

int ScribbleArea::predictNumber(tensorflow::Tensor& tensorVector) {
    Scope scope = Scope::NewRootScope();
    ClientSession session(scope);
    auto div = Sub(scope, 1.0f, Div(scope, tensorVector, {255.f}));
    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({div}, &outputs));
    imageToPredict = make_shared<Tensor>(outputs[0]);
    Helper::printImageInConsole(*imageToPredict);
    Tensor predicted = model->predict(*imageToPredict);
    Tensor predictedClass = Helper::calculatePredictedClass(predicted);
    int predictedChar = predictedClass.flat<int64>()(0); // your code for flat<int64>
    std::cout << "Predicted char " << predictedChar << std::endl;
    return predictedChar;
}

void ScribbleArea::drawCharOnTextLayer(int x, int y, int width, int height, int predictedChar) {
    QString charText = QString::number(predictedChar);
    QPainter painter(&textLayer);
    painter.setFont(QFont("Arial", 50)); // Set the font size large enough to fill the ScribbleArea
    painter.drawText(QRect(x, y, width, height), Qt::AlignCenter, charText);
    update();
}

std::tuple<int, int, int, int> ScribbleArea::calculateDimensionsForExtraction() {
    const int buffer = 50;
    int x = x_min - buffer;
    int y = y_min - buffer;
    int width = x_max - x + buffer;
    int height = y_max - y + buffer;

    if ((x + width) > handwritingLayer.width()) {
        width = handwritingLayer.width() - x;
    }
    if (y + height > handwritingLayer.height()) {
        height = handwritingLayer.height() - y;
    }
    if (x < 0) {
        x = 0;
    }
    if (y < 0) {
        y = 0;
    }
    return {x, y, width, height};
}

void ScribbleArea::trainOnWrittenChar(int expectedNumber) {
    model->retrain(*imageToPredict, expectedNumber);
}

cv::Mat ScribbleArea::QImageToCvMat(const QImage &image) {
    cv::Mat mat;
    if (image.format() == QImage::Format_RGB32 || image.format() == QImage::Format_ARGB32) {
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    } else {
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    }
    return mat.clone();
}

//CvMatToQImage
QImage ScribbleArea::cvMatToQImage(const cv::Mat &inMat) {
    switch (inMat.type()) {
        case CV_8UC1:
            return QImage((const uchar *) inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8);
        case CV_8UC3:
            return QImage((const uchar *) inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888);
        case CV_8UC4:
            return QImage((const uchar *) inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_ARGB32);
        default:
            return QImage();
    }
}

void ScribbleArea::boundingBox() {
        QPainter painter(&textLayer);
        painter.setPen(QPen(QColor(Qt::red), 3, Qt::DotLine, Qt::RoundCap, Qt::RoundJoin));
        QPoint point(x_min, y_min);
        QPoint po(x_max, y_max);
        QRect rect1(point, po);
        painter.drawRect(rect1);
        update();
}


void ScribbleArea::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    QRect dirtyRect = event->rect();
    painter.drawImage(dirtyRect, handwritingLayer, dirtyRect);
    painter.drawImage(0, 0, textLayer);
}


void ScribbleArea::resizeEvent(QResizeEvent *event) {
    if(width() > handwritingLayer.width() || height() > handwritingLayer.height()) {
        int newWidth = qMax(width() + 128, handwritingLayer.width());
        int newHeight = qMax(height() + 128, handwritingLayer.height());
        resizeImage(&handwritingLayer, QSize(newWidth, newHeight), Qt::white);
        resizeImage(&textLayer, QSize(newWidth, newHeight), Qt::transparent);
        update();
    }
    QWidget::resizeEvent(event);
}

void ScribbleArea::drawLineTo(const QPoint &endPoint) {
    QPainter painter(&handwritingLayer);
    painter.setPen(QPen(myPenColor, myPenWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint, endPoint);
    modified = true;
    int rad = (myPenWidth / 2) + 2;
    update(QRect(lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad));
    lastPoint = endPoint;
}

void ScribbleArea::resizeImage(QImage *image, const QSize &newSize, QColor color) {
    if (image->size() == newSize) {
        return;
    }
    QImage newImage(newSize, QImage::Format_ARGB32);
    newImage.fill(color);
    QPainter painter(&newImage);
    painter.drawImage(QPoint(0, 0), *image);
    *image = newImage;
}

void ScribbleArea::transform(QImage *image) {
    QLabel label;
    label.setPixmap(QPixmap::fromImage(image->transformed(QTransform().scale(2,2))));
    label.show();
}

void ScribbleArea::print() {
#if QT_CONFIG(printdialog)
    QPrinter printer(QPrinter::HighResolution);
    QPrintDialog printDialog(&printer, this);
    if (printDialog.exec() == QDialog::Accepted) {
        QPainter painter(&printer);
        QRect rect = painter.viewport();
        QSize size = textLayer.size();
        size.scale(rect.size(), Qt::KeepAspectRatio);
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height());
        painter.setWindow(textLayer.rect());
        painter.drawImage(0, 0, textLayer);
    }
#endif
}

void ScribbleArea::convertImageFormat(QImage &image) {
    if(image.format() != QImage::Format_ARGB32 && image.format() != QImage::Format_ARGB32_Premultiplied) {
        image = image.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    }
}

void ScribbleArea::makeWhitePixelsTransparent(QImage &image) {
    for(int y = 0 ; y < image.height() ; y++) {
        for(int x = 0 ; x < image.width() ; x++){
            QColor color(image.pixel(x,y));
            // Check for full white, i.e., R=G=B=255 and alpha=255
            if(color.red() == 255 && color.green() == 255 && color.blue() == 255) {
                color.setAlpha(0);  // make it transparent
                image.setPixelColor(x, y, color);
            }
        }
    }
}

void ScribbleArea::makeTransparentPixelsWhite(QImage *image) {
    for(int y = 0 ; y < image->height() ; y++) {
        for(int x = 0 ; x < image->width() ; x++){
            QColor color(image->pixel(x,y));
            // Check for full white, i.e., R=G=B=255 and alpha=255
            if(color.red() == 255 && color.green() == 255 && color.blue() == 255) {
                color.setAlpha(255);  // make it transparent
                image->setPixelColor(x, y, color);
            }
        }
    }
}