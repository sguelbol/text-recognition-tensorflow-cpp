#include "../headers/GraphLogger.h"
#include <filesystem>
#include "tensorflow/core/summary/summary_file_writer.h"

void GraphLogger::logGraph(Scope& scope) {
    GraphDef graph;
    scope.ToGraphDef(&graph);
    std::filesystem::path cwd = std::filesystem::current_path().parent_path();
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, cwd.string() + "/logs/", ".img-graph", Env::Default(), &w));
    TF_CHECK_OK(w->WriteGraph(0, std::make_unique<GraphDef>(graph)));
}
