
#import "FaceTracker.h"

#include  "mediapipe/framework/formats/detection.pb.h"

#import "mediapipe/objc/MPPGraph.h"
#import "mediapipe/objc/MPPCameraInputSource.h"
#import "mediapipe/objc/MPPLayerRenderer.h"
//#include "mediapipe/framework/formats/landmark.pb.h"

static NSString* const kGraphName = @"face_detection_mobile_gpu";
static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
//cpu icin olan face_detections
//grpu icin olan detections
static const char* kDetectionsOutputStream = "face_detections";

@interface FaceTracker() <MPPGraphDelegate>
@property(nonatomic) MPPGraph* mediapipeGraph;
@end
@implementation FaceTracker

#pragma mark - UIViewController methods

- (void)dealloc {
    self.mediapipeGraph.delegate = nil;
    [self.mediapipeGraph cancel];
    // Ignore errors since we're cleaning up.
    [self.mediapipeGraph closeAllInputStreamsWithError:nil];
    [self.mediapipeGraph waitUntilDoneWithError:nil];
}


+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
    // Load the graph config resource.
    NSError* configLoadError = nil;
    NSBundle* bundle = [NSBundle bundleForClass:[self class]];
    if (!resource || resource.length == 0) {
        return nil;
    }
    NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
    NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
    if (!data) {
        NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
        return nil;
    }

    // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
    mediapipe::CalculatorGraphConfig config;
    config.ParseFromArray(data.bytes, data.length);

    // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
    MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];

//    [newGraph setSidePacket:(mediapipe::MakePacket<int>(kNumHands))
//                               named:kNumHandsInputSidePacket];

    [newGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
    [newGraph addFrameOutputStream:kDetectionsOutputStream outputPacketType:MPPPacketTypeRaw];
    return newGraph;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
        self.mediapipeGraph.delegate = self;
        // Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
        self.mediapipeGraph.maxFramesInFlight = 2;
    }
    return self;
}

- (void)startGraph {
    // Start running self.mediapipeGraph.
    NSError* error;
    if (![self.mediapipeGraph startWithError:&error]) {
        NSLog(@"Failed to start graph: %@", error);
    }
}
//
//- (void)viewDidLoad {
//  [super viewDidLoad];
//
//  [self.mediapipeGraph addFrameOutputStream:kDetectionsOutputStream
//                           outputPacketType:MPPPacketTypeRaw];
//}

#pragma mark - MPPGraphDelegate methods
- (void)mediapipeGraph:(MPPGraph*)graph
  didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
            fromStream:(const std::string&)streamName {
      if (streamName == kOutputStream) {
          [_delegate faceTracker: self didOutputPixelBuffer: pixelBuffer];
      }
}
// Receives a raw packet from the MediaPipe graph. Invoked on a MediaPipe worker thread.
- (void)mediapipeGraph:(MPPGraph*)graph
     didOutputPacket:(const ::mediapipe::Packet&)packet
          fromStream:(const std::string&)streamName {


  //face landmarks poins
  if (streamName == kDetectionsOutputStream)
  {
        if (packet.IsEmpty())
        {
          NSLog(@"[TS:%lld] No face landmarks", packet.Timestamp().Value());
          return;
        }

    const auto& multi_fac_dect = packet.Get<std::vector<::mediapipe::Detection>>();

          NSLog(@"[TS:%lld] Number of face instances with rects: %lu", packet.Timestamp().Value(),
            multi_fac_dect.size());

    NSMutableArray *faceArray = [[NSMutableArray alloc] init];

          for (int face_index = 0; face_index < multi_fac_dect.size(); ++face_index)
          {
            const auto& location_data = multi_fac_dect[face_index].location_data();

              const auto& keypoints = location_data.relative_keypoints();
              NSLog(@"\tNumber of landmarks for face[%d]: %d", face_index, keypoints.size());

              for (int i = 0; i < keypoints.size(); ++i)
              {

                  const auto& keypoint = keypoints[i];
                  NSLog(@"\t\tFace Landmark[%d]: (%f, %f)", i, keypoint.x(),
                        keypoint.y());
                NSDictionary *landmark = @{@"face": [NSNumber numberWithInt:face_index],
                                      @"landmarkIndex":[NSNumber numberWithInt:i],
                                      @"x":[NSNumber numberWithFloat: keypoint.x()],
                                      @"y":[NSNumber numberWithFloat: keypoint.y()],
                                     };
                [faceArray addObject:landmark];

              }

          }
    
    [_delegate faceTracker: self didOutputLandmarks: faceArray];
      
    }
}




- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer {
    [self.mediapipeGraph sendPixelBuffer:imageBuffer
                              intoStream:kInputStream
                              packetType:MPPPacketTypePixelBuffer];
}



@end


