#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace ir {

lazy_tensors::Shape GetShapeFromTsOutput(const ir::Output& output) {
  if (auto tsnode = dynamic_cast<const TsNode*>(output.node)) {
    return tsnode->shape(output.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

lazy_tensors::Shape GetShapeFromTsValue(const ir::Value& value) {
  if (auto tsnode = dynamic_cast<const TsNode*>(value.node.get())) {
    return tsnode->shape(value.index);
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

lazy_tensors::Shape GetShapeFromTsNode(const ir::Node& node) {
  if (auto tsnode = dynamic_cast<const TsNode*>(&node)) {
    return tsnode->shape();
  }
  throw std::runtime_error("Expected TsNode but could not dynamic cast");
}

void TsNodeSetShapeDeferred(
    NodePtr node, const std::function<lazy_tensors::Shape()>& shape_fn) {
  if (auto tsnode = std::dynamic_pointer_cast<TsNode>(node)) {
    tsnode->SetShapeDeferred(shape_fn);
  } else {
    throw std::runtime_error("Expected TsNode but could not dynamic cast");
  }
}

lazy_tensors::hash_t OperandHashes(const OpList& operands,
                                   const lazy_tensors::hash_t& seed) {
  lazy_tensors::hash_t hash = seed;
  for (auto& operand : operands) {
    hash = lazy_tensors::util::HashCombine(hash, operand.hash());
  }
  return hash;
}

// TODO(whc) rename:
// hash_seed is a misnomer here; it's not really a 'seed', so much as
// the partial node-hash computed by the derived class, typically over any
// scalar constants
TsNode::TsNode(OpKind op, OpList operands, lazy_tensors::Shape shape,
               size_t num_outputs, lazy_tensors::hash_t hash_seed)
    : Node(
          op, operands, num_outputs,
          // TODO(WHC) this is inefficient (having to compute node_hash twice
          // since I can't call hash() yet) so probably move dag_hash
          // initialization to a separate function?
          /* node_hash */ lazy_tensors::util::HashCombine(op.hash(), hash_seed),
          /* dag_hash */
          OperandHashes(operands,
                        lazy_tensors::util::HashCombine(op.hash(), hash_seed))),
      shape_(shape) {}

TsNode::TsNode(OpKind op, OpList operands,
               const std::function<lazy_tensors::Shape()>& shape_fn,
               size_t num_outputs, lazy_tensors::hash_t hash_seed)
    : TsNode(op, operands, lazy_tensors::Shape(), num_outputs, hash_seed) {
  shape_ = GetOpShape(shape_fn);
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               lazy_tensors::hash_t hash_seed)
    : TsNode(op, operands, lazy_tensors::Shape(), num_outputs, hash_seed){}

void TsNode::SetShapeDeferred(
    const std::function<lazy_tensors::Shape()>& shape_fn) {
  shape_ = GetOpShape(shape_fn);
}

TsNode::TsNode(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
               lazy_tensors::hash_t hash_seed)
    : Node(op, num_outputs,
           /* node_hash */ GetOpHash(op, shape, hash_seed)),
      shape_(shape) {
}

const lazy_tensors::Shape& TsNode::shape() const { return shape_; }

const lazy_tensors::Shape& TsNode::shape(size_t output_index) const {
  if (shape_.IsTuple()) {
    return shape_.tuple_shapes(output_index);
  }
  LTC_CHECK_EQ(output_index, 0);
  return shape_;
}

using ShapeCache =
    lazy_tensors::util::Cache<lazy_tensors::hash_t, lazy_tensors::Shape,
                              lazy_tensors::util::HashReducer>;

ShapeCache* GetShapeCache() {
  static lazy_tensors::int64 shape_cache_size =
      lazy_tensors::sys_util::GetEnvInt("LTC_IR_SHAPE_CACHE_SIZE", 4096);
  static ShapeCache* cache = new ShapeCache(shape_cache_size);
  return cache;
}

lazy_tensors::Shape TsNode::GetOpShape(
    const std::function<lazy_tensors::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(),
                             std::make_shared<lazy_tensors::Shape>(shape_fn()));
  }
  return *shape;
}

std::string TsNode::ToString() const {
  std::stringstream ss;
  ss << shape() << " " << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata().scope.empty()) {
    ss << ", scope=" << metadata().scope;
  }
  EmitShortFrameInfo(ss, metadata().frame_info);
  return ss.str();
}

lazy_tensors::hash_t TsNode::GetOpHash(OpKind op,
                                       const lazy_tensors::Shape& shape,
                                       lazy_tensors::hash_t hash_seed) {
  if (lazy_tensors::Shape::IsDynamicMode()) {
    lazy_tensors::hash_t h = lazy_tensors::util::HashCombine(
        op.hash(), lazy_tensors::util::Hash(shape.rank()));
    return lazy_tensors::util::HashCombine(h, hash_seed);
  }
  lazy_tensors::hash_t h = lazy_tensors::util::HashCombine(
      op.hash(), lazy_tensors::util::Hash(shape.ToString()));
  return lazy_tensors::util::HashCombine(h, hash_seed);
}

}  // namespace ir
}  // namespace torch_lazy_tensors