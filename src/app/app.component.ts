import { Component, DestroyRef, ElementRef, ViewChild } from '@angular/core';
import { simd } from 'wasm-feature-detect';

declare var createTFLiteModule: Function;
declare var createTFLiteSIMDModule: Function;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

  @ViewChild('videoEl') video!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasEl') canvas!: ElementRef<HTMLCanvasElement>;

  private readonly options = {
    height: 144,
    width: 256,
    blurValue: 15
  };
  private readonly segmentationPixelCount = this.options.width * this.options.height;

  private tflite: any;
  private animationFrame?: number;

  private segmentationMask!: ImageData;
  private segmentationMaskCanvas!: HTMLCanvasElement;
  private segmentationMaskCtx!: CanvasRenderingContext2D | null;
  private outputCanvasContext?: CanvasRenderingContext2D | null;
  private mediaStream?: MediaStream;

  constructor(destroy: DestroyRef) {
    destroy.onDestroy(() => this.cancelCanvas());
    simd().then(issimd => {
      const script = document.createElement('script');
      if (issimd) {
        script.src = 'assets/tflite/tflite-simd.js';
      } else {
        script.src = 'assets/tflite/tflite.js';
      }
      document.body.append(script);
      script.onload = () => this.init(issimd ? createTFLiteSIMDModule : createTFLiteModule);
    });
  }

  async init(createModule: Function): Promise<void> {
    this.tflite = await createModule();
    let modelBuffer: ArrayBuffer;

    const modelResponse = await fetch('/assets/tflite/selfie_segmentation_landscape.tflite');

    if (!modelResponse.ok) {
      throw new Error('Failed to download tflite model!');
    }

    modelBuffer = await modelResponse.arrayBuffer();

    this.tflite.HEAPU8.set(new Uint8Array(modelBuffer), this.tflite._getModelBufferMemoryOffset());
    this.tflite._loadModel(modelBuffer.byteLength);

    this.segmentationMask = new ImageData(this.options.width, this.options.height);
    this.segmentationMaskCanvas = document.createElement('canvas');
    this.segmentationMaskCanvas.width = this.options.width;
    this.segmentationMaskCanvas.height = this.options.height;
    this.segmentationMaskCtx = this.segmentationMaskCanvas.getContext('2d');
  }

  startCam() {
    this.cancelCanvas();
    navigator.mediaDevices
      .getUserMedia({
        video: {facingMode: 'user', width: 640, height: 360}
      })
      .then((mediaStream) => {
        this.video.nativeElement.srcObject = mediaStream;
        this.mediaStream = mediaStream;
        this.video.nativeElement.onloadedmetadata = () => {
          this.video.nativeElement.play();
        };
        this.outputCanvasContext = this.canvas.nativeElement.getContext('2d');
        this.startCanvas();
      });
  }

  stopCam(): void {
    this.video.nativeElement.remove();
    this.canvas.nativeElement.remove();
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.cancelCanvas();
  }

  private resizeSource() {
    this.segmentationMaskCtx?.drawImage(
      this.video.nativeElement,
      0,
      0,
      this.video.nativeElement.videoWidth,
      this.video.nativeElement.videoHeight,
      0,
      0,
      this.options.width,
      this.options.height
    );

    const imageData = this.segmentationMaskCtx?.getImageData(
      0,
      0,
      this.options.width,
      this.options.height
    );
    const inputMemoryOffset = this.tflite._getInputMemoryOffset() / 4;

    for (let i = 0; i < this.segmentationPixelCount; i++) {
      this.tflite.HEAPF32[inputMemoryOffset + (i * 3)] = Number(imageData?.data[i * 4]) / 255;
      this.tflite.HEAPF32[inputMemoryOffset + (i * 3) + 1] = Number(imageData?.data[(i * 4) + 1]) / 255;
      this.tflite.HEAPF32[inputMemoryOffset + (i * 3) + 2] = Number(imageData?.data[(i * 4) + 2]) / 255;
    }
  }

  private findPerson() {
    this.tflite._runInference();
    const outputMemoryOffset = this.tflite._getOutputMemoryOffset() / 4;

    for (let i = 0; i < this.segmentationPixelCount; i++) {
      const person = this.tflite.HEAPF32[outputMemoryOffset + i];

      // Sets only the alpha component of each pixel.
      this.segmentationMask.data[(i * 4) + 3] = 255 * person;
    }
    this.segmentationMaskCtx?.putImageData(this.segmentationMask, 0, 0);
  }

  private runPostProcessing() {
    if (!this.outputCanvasContext) {
      return;
    }

    const height = this.video.nativeElement.videoHeight;
    const width = this.video.nativeElement.videoWidth;
    this.canvas.nativeElement.height = height;
    this.canvas.nativeElement.width = width;
    this.outputCanvasContext.globalCompositeOperation = 'copy';

    this.outputCanvasContext.filter = 'blur(4px)';
    this.outputCanvasContext?.drawImage(
      this.segmentationMaskCanvas,
      0,
      0,
      this.options.width,
      this.options.height,
      0,
      0,
      this.video.nativeElement.videoWidth,
      this.video.nativeElement.videoHeight
    );
    this.outputCanvasContext.globalCompositeOperation = 'source-in';
    this.outputCanvasContext.filter = 'none';

    // Draw the foreground video.
    this.outputCanvasContext?.drawImage(this.video.nativeElement, 0, 0);

    // Draw the background.
    this.outputCanvasContext.globalCompositeOperation = 'destination-over';
    this.outputCanvasContext.filter = `blur(${this.options.blurValue}px)`;
    this.outputCanvasContext?.drawImage(this.video.nativeElement, 0, 0);
  }

  private startCanvas(): void {
    this.animationFrame = requestAnimationFrame(() => {
      this.resizeSource();
      this.findPerson();
      this.runPostProcessing();
      this.startCanvas();
    });
  }

  private cancelCanvas(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
  }
}
