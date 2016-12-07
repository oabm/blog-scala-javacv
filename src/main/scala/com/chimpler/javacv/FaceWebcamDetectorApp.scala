package com.chimpler.javacv

import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier
import org.bytedeco.javacpp.{opencv_core, opencv_imgproc}
import org.bytedeco.javacv.FrameGrabber.ImageMode
import org.bytedeco.javacv.{CanvasFrame, Frame, OpenCVFrameConverter, OpenCVFrameGrabber}

import org.bytedeco.javacpp.opencv_imgproc.CvFont

/**
 * Created by chimpler on 7/13/14.
 */
object FaceWebcamDetectorApp extends App {

  // holder for a single detected face: contains face rectangle and the two eye rectangles inside
  case class Face(id: Int, faceRect: Rect, leftEyeRect: Rect, rightEyeRect: Rect)

  // we need to clone the rect because openCV is recycling rectangles created by the detectMultiScale method
  private def cloneRect(rect: Rect): Rect = {
    new Rect(rect.x, rect.y, rect.width, rect.height)
  }

  class FaceDetector() {
    // read the haar classifier xml files for face, left eye and right eye
    val faceXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_frontalface_alt.xml").getPath
    val faceCascade = new CascadeClassifier(faceXml)

    val leftEyeXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_mcs_lefteye_alt.xml").getPath
    val leftEyeCascade = new CascadeClassifier(leftEyeXml)

    val rightEyeXml = FaceWebcamDetectorApp.getClass.getClassLoader.getResource("haarcascade_mcs_righteye_alt.xml").getPath
    val rightEyeCascade = new CascadeClassifier(rightEyeXml)

    def detect(greyMat: Mat): Seq[Face] = {
      val faceRects = new RectVector()
      faceCascade.detectMultiScale(greyMat, faceRects)
      for(i <- 0 until faceRects.limit().toInt) yield {
        val faceRect = faceRects.get(i)

        // the left eye should be in the top-left quarter of the face area
        val leftFaceMat = new Mat(greyMat, new Rect(faceRect.x, faceRect.y, faceRect.width() / 2, faceRect.height() / 2))
        val leftEyeRect = new RectVector()
        leftEyeCascade.detectMultiScale(leftFaceMat, leftEyeRect)

        // the right eye should be in the top-right quarter of the face area
        val rightFaceMat = new Mat(greyMat, new Rect(faceRect.x + faceRect.width() / 2, faceRect.y, faceRect.width() / 2, faceRect.height() / 2))
        val rightEyeRect = new RectVector()
        rightEyeCascade.detectMultiScale(rightFaceMat, rightEyeRect)
        if (rightEyeRect.size() > 0 && leftEyeRect.size() > 0) {
          Face(i, cloneRect(faceRect), cloneRect(leftEyeRect.get(0)), cloneRect(rightEyeRect.get(0)))
        } else {
          Face(i, new Rect(0, 0, 10, 10), new Rect(5, 5, 5, 5), new Rect(10, 10, 5, 5))
        }
      }
    }
  }

  val canvas = new CanvasFrame("Webcam")

  val faceDetector = new FaceDetector
  //  //Set Canvas frame to close on exit
  canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE)

  //Declare FrameGrabber to import output from webcam
  val grabber = new OpenCVFrameGrabber(0)
  grabber.setImageWidth(640)
  grabber.setImageHeight(480)
  grabber.setBitsPerPixel(CV_8U)
  grabber.setImageMode(ImageMode.COLOR)
  grabber.start()

  var lastRecognitionTime = 0L
  val cvFont = new CvFont()
  cvFont.hscale(0.6f)
  cvFont.vscale(0.6f)
  cvFont.font_face(FONT_HERSHEY_SIMPLEX)

  val converterToMat = new OpenCVFrameConverter.ToMat();

  var mat: Mat = new Mat(640, 480, CV_8UC3)
  val greyMat: Mat = new Mat(640, 480, CV_8U)
  var faces: Seq[Face] = Nil
  while (true) {
    val img: Frame = grabber.grab()
//    cvMirror(img, img, 1)

    // run the recognition every 200ms to not use too much CPU
    if (System.currentTimeMillis() - lastRecognitionTime > 200) {
      mat = converterToMat.convert(img)
      opencv_imgproc.cvtColor(mat, greyMat, opencv_imgproc.CV_BGR2GRAY, 1)
      opencv_imgproc.equalizeHist(greyMat, greyMat)
      faces = faceDetector.detect(greyMat)
      lastRecognitionTime = System.currentTimeMillis()
    }

    // draw the face rectangles with the eyes and caption
    for(f <- faces) {
      // draw the face rectangle
      opencv_imgproc.rectangle(mat,
        new opencv_core.Point(f.faceRect.x, f.faceRect.y),
        new opencv_core.Point(f.faceRect.x + f.faceRect.width, f.faceRect.y + f.faceRect.height),
        new Scalar(255, 0, 0, 0)
      )

      // draw the left eye rectangle
      opencv_imgproc.rectangle(mat,
        new opencv_core.Point(f.faceRect.x + f.leftEyeRect.x, f.faceRect.y + f.leftEyeRect.y),
        new opencv_core.Point(f.faceRect.x + f.leftEyeRect.x + f.leftEyeRect.width, f.faceRect.y + f.leftEyeRect.y + f.leftEyeRect.height),
        new Scalar(0, 0, 255, 0)
      )

      // draw the right eye rectangle
      opencv_imgproc.rectangle(mat,
        new opencv_core.Point(f.faceRect.x + f.faceRect.width / 2 + f.rightEyeRect.x, f.faceRect.y + f.rightEyeRect.y),
        new opencv_core.Point(f.faceRect.x + f.faceRect.width / 2 + f.rightEyeRect.x + f.rightEyeRect.width, f.faceRect.y + f.rightEyeRect.y + f.rightEyeRect.height),
        new Scalar(0, 255, 0, 0)
      )

      // draw the face number
//      val cvPoint = opencv_core.cvPoint(f.faceRect.x, f.faceRect.y - 20)
//      cvPutText(img, s"Face ${f.id}", cvPoint, cvFont, AbstractCvScalar.RED)
    }
    canvas.showImage(converterToMat.convert(mat))
  }

}
