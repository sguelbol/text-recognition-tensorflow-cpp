#include "../headers/GraphLogger.h"
#include <filesystem>
#include "tensorflow/core/summary/summary_file_writer.h"

/**
 * @brief Log the graph into a summary file, which is necessary for visualization with the Tensorboard of Tensorflow.
 *
 * This method takes a Scope object and logs the graph to a summary file, which is necessary for visualization with the
 * Tensorboard of Tensorflow. The graph stored in the scope is first converted to a GraphDef and
 * the summary file is written which is used to vizualize the graph in tensorboard.
 *
 * @param scope The Scope object containing the graph to be logged.
 */
void GraphLogger::logGraph(Scope& scope) {
    GraphDef graph;
    scope.ToGraphDef(&graph);
    std::filesystem::path cwd = std::filesystem::current_path().parent_path();
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, cwd.string() + "/logs/", ".img-graph", Env::Default(), &w));
    TF_CHECK_OK(w->WriteGraph(0, std::make_unique<GraphDef>(graph)));
}
