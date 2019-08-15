/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

Status ClusterScoping(std::unique_ptr<Graph>* graph) {
  FixupSourceAndSinkEdges(graph->get());

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  opt_options.flib_def = &flib_def;
  SessionOptions session_options;
  session_options.env = Env::Default();
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  opt_options.session_options = &session_options;

  ClusterScopingPass pass;
  return pass.Run(opt_options);
}

absl::flat_hash_map<string, string> GetXlaScopes(const Graph& graph) {
  absl::flat_hash_map<string, string> scopes;
  for (Node* node : graph.nodes()) {
    string scope;
    if (GetNodeAttr(node->attrs(), kXlaAutoJitScopeAttr, &scope).ok()) {
      scopes[node->name()] = scope;
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "_XlaScopes:";
    for (const auto& p : scopes) {
      VLOG(2) << " " << p.first << " -> " << p.second;
    }
  }
  return scopes;
}

Node* BuildStageNode(GraphDefBuilder& builder, string name,
                     std::initializer_list<DataType> dtypes,
                     gtl::ArraySlice<ops::NodeOut> values) {
  auto opts =
      builder.opts().WithName(std::move(name)).WithAttr("dtypes", dtypes);
  if (opts.HaveError()) {
    return nullptr;
  }

  NodeBuilder node_builder(name, "Stage", opts.op_registry());
  node_builder.Input(values);
  return opts.FinalizeBuilder(&node_builder);
}

TEST(XlaCompilationTest, StagePipelinePreserved) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    // Graph:
    //       b
    //       |
    //       v
    // a -> add0 (ClusterX) -> relu0 (ClusterX) -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 (ClusterY) -> relu1 (ClusterY)
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("a")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::SourceOp("Const", builder.opts()
                                         .WithName("b")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* unstage = ops::SourceOp(
        "Unstage",
        builder.opts().WithName("unstage").WithAttr("dtypes", {DT_FLOAT}));

    Node* add0 = ops::BinaryOp("Add", a, b, builder.opts().WithName("add0"));
    Node* add1 =
        ops::BinaryOp("Add", unstage, b, builder.opts().WithName("add1"));
    Node* relu0 = ops::UnaryOp("Relu", add0, builder.opts().WithName("relu0"));
    ops::UnaryOp("Relu", add1, builder.opts().WithName("relu1"));
    BuildStageNode(builder, "stage", {DT_FLOAT}, {relu0});

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ClusterScoping(&graph));

  auto scopes = GetXlaScopes(*graph);
  EXPECT_NE(scopes["add0"], scopes["add1"]);
  EXPECT_EQ(scopes["add0"], scopes["relu0"]);
  EXPECT_EQ(scopes["add1"], scopes["relu1"]);
}

TEST(XlaCompilationTest, StagePipelinePreservedAndInitialScopesRespected) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    // Graph:
    //       b
    //       |
    //       v
    // a -> add0 (ClusterA) -> relu0 (ClusterB) -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 (ClusterC) -> relu1 (ClusterD)
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("a")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::SourceOp("Const", builder.opts()
                                         .WithName("b")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* unstage = ops::SourceOp(
        "Unstage",
        builder.opts().WithName("unstage").WithAttr("dtypes", {DT_FLOAT}));

    // Intentionally give add0 and add1 the same initial scope but they should
    // be separated by the ClusterScopingPass.
    Node* add0 =
        ops::BinaryOp("Add", a, b, builder.opts().WithName("add0").WithAttr(
                                       kXlaAutoJitScopeAttr, "ClusterA"));
    Node* add1 = ops::BinaryOp("Add", unstage, b,
                               builder.opts().WithName("add1").WithAttr(
                                   kXlaAutoJitScopeAttr, "ClusterA"));
    Node* relu0 =
        ops::UnaryOp("Relu", add0, builder.opts().WithName("relu0").WithAttr(
                                       kXlaAutoJitScopeAttr, "ClusterB"));
    ops::UnaryOp("Relu", add1, builder.opts().WithName("relu1").WithAttr(
                                   kXlaAutoJitScopeAttr, "ClusterD"));
    BuildStageNode(builder, "stage", {DT_FLOAT}, {relu0});

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ClusterScoping(&graph));

  auto scopes = GetXlaScopes(*graph);
  EXPECT_NE(scopes["add0"], scopes["add1"]);
  EXPECT_NE(scopes["add0"], scopes["relu0"]);
  EXPECT_NE(scopes["add1"], scopes["relu1"]);
}

}  // namespace
}  // namespace tensorflow