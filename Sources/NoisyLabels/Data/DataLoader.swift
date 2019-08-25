// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import Logging
import Progress

#if os(Linux)
import FoundationNetworking
#endif

/// Contains helpers for dataset loaders.
public protocol DataLoader {
  associatedtype Instance
  associatedtype Predictor
  associatedtype Label

  /// Downloads the file at `url` to `path`, if `path` does not exist.
  ///
  /// - Parameters:
  ///   - from: URL to download data from.
  ///   - to: Destination file path.
  ///
  /// - Returns: Boolean value indicating whether a download was
  ///     performed (as opposed to not needed).
  func maybeDownload(from url: URL, to destination: URL) throws

  func load(withFeatures: Bool) throws -> Data<Instance, Predictor, Label>
}

public extension DataLoader {
  func maybeDownload(from url: URL, to destination: URL) throws {
    if !FileManager.default.fileExists(atPath: destination.path) {
      // Create any potentially missing directories.
      try FileManager.default.createDirectory(
        atPath: destination.deletingLastPathComponent().path,
        withIntermediateDirectories: true)

      // Create the URL session that will be used to download the dataset.
      let semaphore = DispatchSemaphore(value: 0)
      let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
      let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

      // Download the data to a temporary file and then copy that file to
      // the destination path.
      logger.info("Downloading \(url).")
      let task = session.downloadTask(with: url)
      task.resume()

      // Wait for the download to finish.
      semaphore.wait()
    }
  }
}

internal class DataDownloadDelegate : NSObject, URLSessionDownloadDelegate {
  let destinationFileUrl: URL
  let semaphore: DispatchSemaphore
  let numBytesFrequency: Int64 

  internal var logCount: Int64 = 0
  internal var progressBar: ProgressBar? = nil

  init(
    destinationFileUrl: URL,
    semaphore: DispatchSemaphore,
    numBytesFrequency: Int64 = 1024 * 1024
  ) {
    self.destinationFileUrl = destinationFileUrl
    self.semaphore = semaphore
    self.numBytesFrequency = numBytesFrequency
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) -> Void {
    if progressBar == nil {
      progressBar = ProgressBar(
        count: Int(totalBytesExpectedToWrite) / (1024 * 1024),
        configuration: [
          ProgressString(string: "Download Progress (MBs):"),
          ProgressIndex(),
          ProgressBarLine(),
          ProgressTimeEstimates()])
    }
    progressBar!.setValue(Int(totalBytesWritten) / (1024 * 1024))
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) -> Void {
    do {
      try FileManager.default.moveItem(at: location, to: destinationFileUrl)
    } catch (let writeError) {
      logger.error("Error writing file \(location.path) : \(writeError)")
    }
    logger.info("Downloaded successfully to \(location.path).")
    semaphore.signal()
  }
}
