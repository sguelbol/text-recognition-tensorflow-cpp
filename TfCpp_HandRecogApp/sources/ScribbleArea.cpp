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
}


// Functions

void ScribbleArea::setModel(const std::shared_ptr<Model> model) {
    this->model = std::move(model);
}

bool ScribbleArea::openImage(const QString &fileName) {
    QImage loadedImage;
    if (!loadedImage.load(fileName)) {
        return false;
    }
    QSize newSize = loadedImage.size().expandedTo(size());
    resizeImage(&loadedImage, newSize);
    image = loadedImage;
    modified = false;
    update();
    return true;
}


bool ScribbleArea::saveImage(const QString &fileName, const char *fileFormat) {
    QImage visibleImage = image;
    qDebug() <<fileFormat;
    resizeImage(&visibleImage, QSize(400, 400));
    if (visibleImage.save(fileName, fileFormat)) {
        modified = false;
        return true;
    } else {
        return false;
    }
}

void ScribbleArea::setPenColor(const QColor &newColor) {
    myPenColor = newColor;
}

void ScribbleArea::setPenWidth(int newWidth) {
    myPenWidth = newWidth;
}

void ScribbleArea::clearImage() {
    image.fill(qRgb(255, 255, 255));
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


void ScribbleArea::keyReleaseEvent(QKeyEvent *event) {
    qDebug() << "Shift Release 2";
    if ((event->key() == Qt::Key_Shift) && !(QApplication::mouseButtons() & Qt::LeftButton)) {
        qDebug() << "Shift Release 3";
        scribbling = false;
        //transform(&image);
        qDebug() << "Release";
        //hier ausgeben
        qDebug() << "Inputs:" << inputs;
        //Code here bounding box
        //boundingBox();
        inputs.clear();
    }
}

void ScribbleArea::mouseMoveEvent(QMouseEvent *event) {
    if((event->buttons() & Qt::LeftButton) && scribbling) {
        drawLineTo(event->pos());

        //hier sammeln
        inputs.insert(inputs.end(), {event->pos().x(), event->pos().y()});

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
        qDebug() << "x_min: " << x_min << " x_max: " << x_max << ";  y_min:" << y_min << "y_max: " << y_max;
    }
}

void ScribbleArea::mouseReleaseEvent(QMouseEvent *event) {
    if(event->button() == Qt::LeftButton && scribbling) {
        if (!(event->modifiers() & Qt::ShiftModifier)) {
            drawLineTo(event->pos());
            scribbling = false;
            qDebug() << "Release";
            //hier ausgeben
            qDebug() << inputs;
            //Code here bounding box
            //boundingBox();
            //image.transformed(QTransform)

            int x = x_min-50;
            int y = y_min-50;
            int width = x_max-x_min+100;
            int height = y_max-y_min+100;
            if ((x+width) > image.width()) {
                width = image.width();
            }
            if (y+height > image.height()) {
                height = image.height();
            }
            if (x < 0) {
                width = 0;
            }
            if (y < 0) {
                height = 0;
            }
            QImage letter = image.copy(x, y, width, height);
            if (!letter.isNull()) {
                transformTo28(&letter);
                letter.save(QString("/home/sguelbol/CodeContext/text_recognition_eink/noteApps/freeWritingApp/cmake-build-debug/untitled.png"), "PNG");
            }
            inputs.clear();
        }

    }
}

void ScribbleArea::transformTo28(QImage *letter) {
    //convert QImage *letter to cv::Mat
    cv::Mat input = QImageToCvMat(*letter);
    cv::Mat src = cv::Mat(28, 28, CV_8UC3);
    cvtColor(input, src, cv::COLOR_BGR2GRAY);
    cv::Mat resizedImage = cv::Mat(28, 28, CV_8UC3);

    cv::resize(src, resizedImage, cv::Size(28, 28), 0,0,cv::INTER_AREA);
    imshow("resized to 28x28", resizedImage);

    /*for (int i = 0; i < resizedImage.rows; ++i) {
        for (int j = 0; j < resizedImage.cols; ++j) {
            std::cout << static_cast<float>(resizedImage.at<uchar>(i, j)) << " ";
        }
        std::cout << std::endl;
    }*/
    tensorflow::Tensor tensorVector(tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 784});
    auto tmap = tensorVector.tensor<float, 2>();

    for (int i = 0; i < resizedImage.rows; ++i) {
        for (int j = 0; j < resizedImage.cols; ++j) {
            tmap(0, 28*i+j) = static_cast<float>(resizedImage.at<uchar>(i, j));
        }
    }
    Scope scope = Scope::NewRootScope();
    ClientSession session(scope);
    auto div = Sub(scope, 1.0f, Div(scope, tensorVector, {255.f}));
    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({div}, &outputs));
    Tensor predicted = model->predict(outputs[0]);
    Tensor cl = Helper::calculatePredictedClass(predicted);
    std::cout << cl.flat<int64>() << std::endl;
    cvMatToQImage(resizedImage).save(QString("/home/sguelbol/CodeContext/text_recognition_eink/noteApps/freeWritingApp/cmake-build-debug/transformed.png"), "PNG");
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
    if (x_max > -1) {
        QPainter painter(&image);
        painter.setPen(QPen(QColor(Qt::red), 3, Qt::DotLine, Qt::RoundCap, Qt::RoundJoin));
        QPoint point(x_min-70, y_min-70);
        //qDebug() << x_min << " y " << y_min;
        QPoint po(x_max+70, y_max+70);
        QRect rect1(point, po);
        painter.drawRect(rect1);
        update();
    }
}


void ScribbleArea::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    QRect dirtyRect = event->rect();
    painter.drawImage(dirtyRect, image, dirtyRect);
}



void ScribbleArea::resizeEvent(QResizeEvent *event) {
    if(width() > image.width() || height() > image.height()) {
        int newWidth = qMax(width() + 128, image.width());
        int newHeight = qMax(height() + 128, image.height());
        resizeImage(&image, QSize(newWidth, newHeight));
        update();
    }
    QWidget::resizeEvent(event);
}

void ScribbleArea::drawLineTo(const QPoint &endPoint) {
    QPainter painter(&image);
    painter.setPen(QPen(myPenColor, myPenWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint, endPoint);
    modified = true;
    int rad = (myPenWidth / 2) + 2;
    update(QRect(lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad));
    lastPoint = endPoint;
}

void ScribbleArea::resizeImage(QImage *image, const QSize &newSize) {
    if (image->size() == newSize) {
        return;
    }

    QImage newImage(newSize, QImage::Format_ARGB32);
    newImage.fill(qRgb(255, 255, 255));
    QPainter painter(&newImage);
    painter.drawImage(QPoint(0, 0), *image);
    *image = newImage;
}

void ScribbleArea::transform(QImage *image) {
    qDebug() << "transform";
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
        QSize size = image.size();
        size.scale(rect.size(), Qt::KeepAspectRatio);
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height());
        painter.setWindow(image.rect());
        painter.drawImage(0, 0, image);
    }
#endif
}