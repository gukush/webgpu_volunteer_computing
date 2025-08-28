// timing.js - Comprehensive timing measurements for volunteer computing system
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';

export class TimingManager {
  constructor() {
    this.activeTasks = new Map(); // taskId -> task timing data
    this.activeChunks = new Map(); // chunkId -> chunk timing data
    this.completedTasks = new Map(); // taskId -> final timing data
    this.storageRoot = process.env.VOLUNTEER_STORAGE || path.join(process.cwd(), 'storage');
    this.ensureStorageDir();
  }

  ensureStorageDir() {
    const timingDir = path.join(this.storageRoot, 'timing');
    if (!fs.existsSync(timingDir)) {
      fs.mkdirSync(timingDir, { recursive: true });
    }
  }

  /**
   * Start timing for a new task
   */
  startTask(taskId, taskMetadata = {}) {
    const taskTiming = {
      taskId,
      startTime: performance.now(),
      startTimestamp: new Date().toISOString(),
      metadata: taskMetadata,
      chunks: new Map(),
      assemblyTimes: [],
      totalTime: null,
      completedAt: null
    };
    
    this.activeTasks.set(taskId, taskTiming);
    console.log(`[TIMING] Started task ${taskId}`);
    return taskTiming;
  }

  /**
   * Start timing for a chunk within a task
   */
  startChunk(taskId, chunkId, chunkMetadata = {}) {
    const taskTiming = this.activeTasks.get(taskId);
    if (!taskTiming) {
      console.warn(`[TIMING] Attempted to start chunk ${chunkId} for unknown task ${taskId}`);
      return null;
    }

    const chunkTiming = {
      chunkId,
      taskId,
      startTime: performance.now(),
      startTimestamp: new Date().toISOString(),
      metadata: chunkMetadata,
      chunkingTime: null,
      clientProcessingTime: null,
      roundTripTime: null,
      completedAt: null
    };

    taskTiming.chunks.set(chunkId, chunkTiming);
    this.activeChunks.set(chunkId, chunkTiming);
    console.log(`[TIMING] Started chunk ${chunkId} for task ${taskId}`);
    return chunkTiming;
  }

  /**
   * Record chunking strategy time for a chunk
   */
  recordChunkingTime(chunkId, chunkingTime) {
    const chunkTiming = this.activeChunks.get(chunkId);
    if (chunkTiming) {
      chunkTiming.chunkingTime = chunkingTime;
      console.log(`[TIMING] Chunking time for ${chunkId}: ${chunkingTime.toFixed(2)}ms`);
    }
  }

  /**
   * Record client processing time for a chunk
   */
  recordClientProcessingTime(chunkId, processingTime) {
    const chunkTiming = this.activeChunks.get(chunkId);
    if (chunkTiming) {
      chunkTiming.clientProcessingTime = processingTime;
      chunkTiming.completedAt = new Date().toISOString();
      console.log(`[TIMING] Client processing time for ${chunkId}: ${processingTime.toFixed(2)}ms`);
    }
  }

  /**
   * Record assembly time for a chunk or task
   */
  recordAssemblyTime(taskId, assemblyTime, chunkId = null) {
    const taskTiming = this.activeTasks.get(taskId);
    if (taskTiming) {
      const assemblyRecord = {
        timestamp: new Date().toISOString(),
        assemblyTime,
        chunkId
      };
      taskTiming.assemblyTimes.push(assemblyRecord);
      console.log(`[TIMING] Assembly time for ${chunkId || 'task'}: ${assemblyTime.toFixed(2)}ms`);
    }
  }

  /**
   * Complete a chunk and calculate round-trip time
   */
  completeChunk(chunkId) {
    const chunkTiming = this.activeChunks.get(chunkId);
    if (chunkTiming) {
      const endTime = performance.now();
      chunkTiming.roundTripTime = endTime - chunkTiming.startTime;
      chunkTiming.completedAt = new Date().toISOString();
      console.log(`[TIMING] Completed chunk ${chunkId}, round-trip time: ${chunkTiming.roundTripTime.toFixed(2)}ms`);
    }
  }

  /**
   * Complete a task and calculate total time
   */
  completeTask(taskId) {
    const taskTiming = this.activeTasks.get(taskId);
    if (taskTiming) {
      const endTime = performance.now();
      taskTiming.totalTime = endTime - taskTiming.startTime;
      taskTiming.completedAt = new Date().toISOString();
      
      // Move to completed tasks
      this.completedTasks.set(taskId, taskTiming);
      this.activeTasks.delete(taskId);
      
      console.log(`[TIMING] Completed task ${taskId}, total time: ${taskTiming.totalTime.toFixed(2)}ms`);
      
      // Generate timing reports
      this.generateTimingReports(taskId);
    }
  }

  /**
   * Generate timing reports for a completed task
   */
  generateTimingReports(taskId) {
    const taskTiming = this.completedTasks.get(taskId);
    if (!taskTiming) return;

    try {
      // Generate per-chunk CSV
      this.generateChunkTimingCSV(taskId);
      
      // Generate task summary CSV
      this.generateTaskSummaryCSV(taskId);
      
      // Generate task descriptor file
      this.generateTaskDescriptor(taskId);
      
    } catch (error) {
      console.error(`[TIMING] Error generating reports for task ${taskId}:`, error);
    }
  }

  /**
   * Generate CSV with per-chunk timing data
   */
  generateChunkTimingCSV(taskId) {
    const taskTiming = this.completedTasks.get(taskId);
    if (!taskTiming) return;

    const csvLines = [
      'task_id,chunk_id,chunking_time_ms,client_processing_time_ms,round_trip_time_ms,start_timestamp,completed_timestamp'
    ];

    for (const [chunkId, chunkTiming] of taskTiming.chunks) {
      const line = [
        taskId,
        chunkId,
        chunkTiming.chunkingTime?.toFixed(2) || '',
        chunkTiming.clientProcessingTime?.toFixed(2) || '',
        chunkTiming.roundTripTime?.toFixed(2) || '',
        chunkTiming.startTimestamp,
        chunkTiming.completedAt || ''
      ].join(',');
      
      csvLines.push(line);
    }

    const csvContent = csvLines.join('\n');
    const filename = `chunk_timing_${taskId}.csv`;
    const filepath = path.join(this.storageRoot, 'timing', filename);
    
    fs.writeFileSync(filepath, csvContent);
    console.log(`[TIMING] Generated chunk timing CSV: ${filepath}`);
  }

  /**
   * Generate CSV with task summary data
   */
  generateTaskSummaryCSV(taskId) {
    const taskTiming = this.completedTasks.get(taskId);
    if (!taskTiming) return;

    // Check if summary CSV exists, create header if not
    const summaryFilepath = path.join(this.storageRoot, 'timing', 'task_summaries.csv');
    let csvContent = '';
    let needsHeader = false;

    if (!fs.existsSync(summaryFilepath)) {
      needsHeader = true;
    }

    if (needsHeader) {
      csvContent = 'task_id,task_description,strategy,framework,parameters,total_time_ms,chunks_count,start_timestamp,completed_timestamp\n';
    }

    // Calculate communication time (inferred)
    const totalChunkingTime = Array.from(taskTiming.chunks.values())
      .reduce((sum, chunk) => sum + (chunk.chunkingTime || 0), 0);
    
    const totalAssemblyTime = taskTiming.assemblyTimes
      .reduce((sum, record) => sum + record.assemblyTime, 0);
    
    const totalClientTime = Array.from(taskTiming.chunks.values())
      .reduce((sum, chunk) => sum + (chunk.clientProcessingTime || 0), 0);
    
    const communicationTime = taskTiming.totalTime - totalChunkingTime - totalAssemblyTime - totalClientTime;

    // Extract parameters from metadata
    const params = this.extractTaskParameters(taskTiming.metadata);
    
    const line = [
      taskId,
      taskTiming.metadata.label || taskTiming.metadata.description || 'Unknown',
      taskTiming.metadata.chunkingStrategy || 'Unknown',
      taskTiming.metadata.framework || 'Unknown',
      JSON.stringify(params),
      taskTiming.totalTime?.toFixed(2) || '',
      taskTiming.chunks.size,
      taskTiming.startTimestamp,
      taskTiming.completedAt || ''
    ].join(',');

    csvContent += line + '\n';
    
    fs.appendFileSync(summaryFilepath, csvContent);
    console.log(`[TIMING] Updated task summary CSV: ${summaryFilepath}`);
  }

  /**
   * Generate task descriptor file with detailed information
   */
  generateTaskDescriptor(taskId) {
    const taskTiming = this.completedTasks.get(taskId);
    if (!taskTiming) return;

    const descriptor = {
      task_id: taskId,
      task_description: taskTiming.metadata.label || taskTiming.metadata.description || 'Unknown',
      strategy: {
        chunking: taskTiming.metadata.chunkingStrategy || 'Unknown',
        assembly: taskTiming.metadata.assemblyStrategy || 'Unknown'
      },
      framework: taskTiming.metadata.framework || 'Unknown',
      parameters: this.extractTaskParameters(taskTiming.metadata),
      timing: {
        total_time_ms: taskTiming.totalTime?.toFixed(2) || 0,
        start_timestamp: taskTiming.startTimestamp,
        completed_timestamp: taskTiming.completedAt || '',
        chunks_count: taskTiming.chunks.size
      },
      performance_breakdown: {
        total_chunking_time_ms: Array.from(taskTiming.chunks.values())
          .reduce((sum, chunk) => sum + (chunk.chunkingTime || 0), 0).toFixed(2),
        total_assembly_time_ms: taskTiming.assemblyTimes
          .reduce((sum, record) => sum + record.assemblyTime, 0).toFixed(2),
        total_client_processing_time_ms: Array.from(taskTiming.chunks.values())
          .reduce((sum, chunk) => sum + (chunk.clientProcessingTime || 0), 0).toFixed(2),
        inferred_communication_time_ms: (taskTiming.totalTime - 
          Array.from(taskTiming.chunks.values()).reduce((sum, chunk) => sum + (chunk.chunkingTime || 0), 0) -
          taskTiming.assemblyTimes.reduce((sum, record) => sum + record.assemblyTime, 0) -
          Array.from(taskTiming.chunks.values()).reduce((sum, chunk) => sum + (chunk.clientProcessingTime || 0), 0)).toFixed(2)
      },
      chunks: Array.from(taskTiming.chunks.values()).map(chunk => ({
        chunk_id: chunk.chunkId,
        chunking_time_ms: chunk.chunkingTime?.toFixed(2) || 0,
        client_processing_time_ms: chunk.clientProcessingTime?.toFixed(2) || 0,
        round_trip_time_ms: chunk.roundTripTime?.toFixed(2) || 0,
        start_timestamp: chunk.startTimestamp,
        completed_timestamp: chunk.completedAt || ''
      }))
    };

    const filename = `task_descriptor_${taskId}.json`;
    const filepath = path.join(this.storageRoot, 'timing', filename);
    
    fs.writeFileSync(filepath, JSON.stringify(descriptor, null, 2));
    console.log(`[TIMING] Generated task descriptor: ${filepath}`);
  }

  /**
   * Extract task parameters from metadata
   */
  extractTaskParameters(metadata) {
    const params = {};
    
    // Common parameters
    if (metadata.matrixSize) params.matrixSize = metadata.matrixSize;
    if (metadata.chunkSize) params.chunkSize = metadata.chunkSize;
    if (metadata.B1) params.B1 = metadata.B1;
    if (metadata.curvesTotal) params.curvesTotal = metadata.curvesTotal;
    if (metadata.curvesPerChunk) params.curvesPerChunk = metadata.curvesPerChunk;
    if (metadata.n) params.n = metadata.n;
    if (metadata.k) params.k = metadata.k;
    if (metadata.batchSize) params.batchSize = metadata.batchSize;
    
    // Input file sizes
    if (metadata.inputRefs) {
      params.inputFiles = metadata.inputRefs.map(ref => ({
        name: ref.name,
        size_bytes: ref.size
      }));
    }
    
    // Framework-specific parameters
    if (metadata.workgroupCount) params.workgroupCount = metadata.workgroupCount;
    if (metadata.entry) params.entry = metadata.entry;
    
    return params;
  }

  /**
   * Get timing statistics for a task
   */
  getTaskTimingStats(taskId) {
    const taskTiming = this.completedTasks.get(taskId);
    if (!taskTiming) return null;

    const chunkTimes = Array.from(taskTiming.chunks.values());
    
    return {
      taskId,
      totalTime: taskTiming.totalTime,
      chunksCount: chunkTimes.length,
      avgChunkingTime: chunkTimes.reduce((sum, chunk) => sum + (chunk.chunkingTime || 0), 0) / chunkTimes.length,
      avgClientTime: chunkTimes.reduce((sum, chunk) => sum + (chunk.clientProcessingTime || 0), 0) / chunkTimes.length,
      avgRoundTripTime: chunkTimes.reduce((sum, chunk) => sum + (chunk.roundTripTime || 0), 0) / chunkTimes.length,
      totalAssemblyTime: taskTiming.assemblyTimes.reduce((sum, record) => sum + record.assemblyTime, 0)
    };
  }

  /**
   * Clean up old timing data
   */
  cleanupOldData(maxAgeHours = 24) {
    const cutoff = Date.now() - (maxAgeHours * 60 * 60 * 1000);
    let cleanedCount = 0;

    for (const [taskId, taskTiming] of this.completedTasks.entries()) {
      if (taskTiming.startTime < cutoff) {
        this.completedTasks.delete(taskId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      console.log(`[TIMING] Cleaned up ${cleanedCount} old timing records`);
    }
  }
}

// Export singleton instance
export const timingManager = new TimingManager();
