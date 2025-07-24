"""
Tree Pruning Adapter for Speculative Streaming

Implements efficient pruning algorithms for speculation trees to maintain
performance while controlling memory usage and computational overhead.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import heapq
from dataclasses import dataclass
import numpy as np


@dataclass
class PruningStats:
    """Statistics for tree pruning operations."""
    nodes_pruned: int = 0
    average_tree_depth: float = 0.0
    average_tree_width: float = 0.0
    pruning_efficiency: float = 0.0


class TreePruningAdapter(nn.Module):
    """
    Adaptive tree pruning for speculation trees in Speculative Streaming.
    
    Uses learned pruning strategies to efficiently maintain speculation trees
    while preserving the most promising speculation paths.
    """
    
    def __init__(
        self,
        hidden_size: int,
        prune_threshold: float = 0.1,
        max_tree_width: int = 8,
        adaptive_pruning: bool = True,
        pruning_strategy: str = "entropy_based",  # "probability", "entropy_based", "learned"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.prune_threshold = prune_threshold
        self.max_tree_width = max_tree_width
        self.adaptive_pruning = adaptive_pruning
        self.pruning_strategy = pruning_strategy
        
        # Learned pruning components
        if pruning_strategy == "learned":
            self.pruning_scorer = nn.Sequential(
                nn.Linear(hidden_size + 3, hidden_size // 2),  # +3 for prob, depth, entropy
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        # Adaptive threshold learning
        if adaptive_pruning:
            self.threshold_adapter = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # Statistics tracking
        self.stats = PruningStats()
        self.reset_stats()
    
    def reset_stats(self):
        """Reset pruning statistics."""
        self.stats = PruningStats()
    
    def compute_node_entropy(self, logits: torch.Tensor) -> float:
        """Compute entropy of token distribution."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy
    
    def compute_pruning_score(self, node, context_embedding: Optional[torch.Tensor] = None) -> float:
        """
        Compute pruning score for a speculation node.
        Higher scores indicate nodes that should be kept.
        """
        if self.pruning_strategy == "probability":
            return node.probability
        
        elif self.pruning_strategy == "entropy_based":
            entropy = self.compute_node_entropy(node.logits)
            # Combine probability and entropy (higher entropy = more uncertainty = lower score)
            return node.probability * np.exp(-entropy / 10.0)
        
        elif self.pruning_strategy == "learned" and context_embedding is not None:
            # Create feature vector for learned pruning
            entropy = self.compute_node_entropy(node.logits)
            features = torch.cat([
                context_embedding.flatten(),
                torch.tensor([node.probability, node.depth, entropy], device=context_embedding.device)
            ])
            
            with torch.no_grad():
                score = self.pruning_scorer(features.unsqueeze(0)).item()
            return score
        
        else:
            # Fallback to probability-based pruning
            return node.probability
    
    def adaptive_threshold(self, context_embedding: torch.Tensor) -> float:
        """Compute adaptive pruning threshold based on context."""
        if not self.adaptive_pruning:
            return self.prune_threshold
        
        with torch.no_grad():
            adaptive_factor = self.threshold_adapter(context_embedding.mean(dim=0)).item()
        
        # Scale base threshold by adaptive factor
        return self.prune_threshold * (0.5 + adaptive_factor)
    
    def prune_branches(
        self, 
        nodes: List, 
        max_width: int,
        context_embedding: Optional[torch.Tensor] = None
    ) -> List:
        """
        Prune speculation tree branches to maintain maximum width.
        
        Args:
            nodes: List of speculation nodes to potentially prune
            max_width: Maximum number of nodes to keep
            context_embedding: Context embedding for adaptive pruning
            
        Returns:
            Pruned list of nodes
        """
        if len(nodes) <= max_width:
            return nodes
        
        # Compute pruning scores for all nodes
        scored_nodes = []
        threshold = self.adaptive_threshold(context_embedding) if context_embedding is not None else self.prune_threshold
        
        for node in nodes:
            score = self.compute_pruning_score(node, context_embedding)
            
            # Apply threshold filtering
            if score >= threshold:
                scored_nodes.append((score, node))
        
        # If threshold filtering leaves too few nodes, relax threshold
        if len(scored_nodes) < max_width // 2:
            scored_nodes = [(self.compute_pruning_score(node, context_embedding), node) for node in nodes]
        
        # Sort by score (descending) and keep top max_width nodes
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        pruned_nodes = [node for _, node in scored_nodes[:max_width]]
        
        # Update statistics
        self.stats.nodes_pruned += len(nodes) - len(pruned_nodes)
        
        return pruned_nodes
    
    def prune_by_depth(self, root_node, max_depth: int) -> None:
        """Prune tree branches that exceed maximum depth."""
        def prune_recursive(node, current_depth):
            if current_depth >= max_depth:
                node.children = []
                return
            
            for child in node.children:
                prune_recursive(child, current_depth + 1)
        
        prune_recursive(root_node, 0)
    
    def smart_tree_pruning(
        self,
        root_node,
        max_total_nodes: int = 50,
        context_embedding: Optional[torch.Tensor] = None
    ) -> None:
        """
        Intelligent tree pruning that maintains the most promising paths.
        
        Uses a combination of breadth-first and depth-first strategies to
        optimize the speculation tree structure.
        """
        # Count current nodes
        total_nodes = self._count_tree_nodes(root_node)
        
        if total_nodes <= max_total_nodes:
            return
        
        # Use priority queue to maintain best nodes across all levels
        priority_queue = []
        node_id = 0
        
        def add_node_to_queue(node, parent_score=1.0):
            nonlocal node_id
            score = self.compute_pruning_score(node, context_embedding) * parent_score
            heapq.heappush(priority_queue, (-score, node_id, node))  # Negative for max-heap
            node_id += 1
        
        # Add all nodes to priority queue
        self._traverse_tree(root_node, add_node_to_queue)
        
        # Keep only the best max_total_nodes
        kept_nodes = set()
        for _ in range(min(max_total_nodes, len(priority_queue))):
            if priority_queue:
                _, _, node = heapq.heappop(priority_queue)
                kept_nodes.add(id(node))
        
        # Prune tree based on kept nodes
        self._prune_tree_by_node_set(root_node, kept_nodes)
        
        # Update statistics
        final_nodes = self._count_tree_nodes(root_node)
        self.stats.nodes_pruned += total_nodes - final_nodes
    
    def _count_tree_nodes(self, root_node) -> int:
        """Count total nodes in tree."""
        count = 1  # Count root
        for child in root_node.children:
            count += self._count_tree_nodes(child)
        return count
    
    def _traverse_tree(self, node, callback, parent_score=1.0):
        """Traverse tree and apply callback to each node."""
        callback(node, parent_score)
        node_score = self.compute_pruning_score(node)
        for child in node.children:
            self._traverse_tree(child, callback, parent_score * node_score)
    
    def _prune_tree_by_node_set(self, node, kept_nodes):
        """Prune tree to keep only nodes in the kept_nodes set."""
        # Filter children
        node.children = [
            child for child in node.children 
            if id(child) in kept_nodes
        ]
        
        # Recursively prune children
        for child in node.children:
            self._prune_tree_by_node_set(child, kept_nodes)
    
    def calculate_tree_metrics(self, root_node) -> Dict[str, float]:
        """Calculate various metrics for the speculation tree."""
        depths = []
        widths_by_level = {}
        
        def collect_metrics(node, depth=0):
            depths.append(depth)
            
            if depth not in widths_by_level:
                widths_by_level[depth] = 0
            widths_by_level[depth] += 1
            
            for child in node.children:
                collect_metrics(child, depth + 1)
        
        collect_metrics(root_node)
        
        return {
            'total_nodes': len(depths),
            'max_depth': max(depths) if depths else 0,
            'average_depth': np.mean(depths) if depths else 0,
            'max_width': max(widths_by_level.values()) if widths_by_level else 0,
            'average_width': np.mean(list(widths_by_level.values())) if widths_by_level else 0,
            'levels': len(widths_by_level),
        }
    
    def optimize_tree_structure(
        self,
        root_node,
        target_metrics: Dict[str, float],
        context_embedding: Optional[torch.Tensor] = None
    ) -> None:
        """
        Optimize tree structure to meet target metrics.
        
        Args:
            root_node: Root of the speculation tree
            target_metrics: Target values for tree metrics
            context_embedding: Context for adaptive decisions
        """
        current_metrics = self.calculate_tree_metrics(root_node)
        
        # Adjust max_tree_width based on current structure
        if 'max_width' in target_metrics:
            target_width = target_metrics['max_width']
            if current_metrics['max_width'] > target_width:
                self._reduce_tree_width(root_node, target_width, context_embedding)
        
        # Adjust max depth
        if 'max_depth' in target_metrics:
            target_depth = target_metrics['max_depth']
            if current_metrics['max_depth'] > target_depth:
                self.prune_by_depth(root_node, int(target_depth))
        
        # Optimize total nodes
        if 'total_nodes' in target_metrics:
            target_total = target_metrics['total_nodes']
            if current_metrics['total_nodes'] > target_total:
                self.smart_tree_pruning(root_node, int(target_total), context_embedding)
    
    def _reduce_tree_width(
        self, 
        root_node, 
        target_width: int, 
        context_embedding: Optional[torch.Tensor] = None
    ):
        """Reduce tree width at each level to target width."""
        def reduce_level_width(node):
            if len(node.children) > target_width:
                node.children = self.prune_branches(
                    node.children, 
                    target_width, 
                    context_embedding
                )
            
            for child in node.children:
                reduce_level_width(child)
        
        reduce_level_width(root_node)
    
    def get_pruning_statistics(self) -> Dict[str, float]:
        """Get current pruning statistics."""
        return {
            'nodes_pruned': self.stats.nodes_pruned,
            'average_tree_depth': self.stats.average_tree_depth,
            'average_tree_width': self.stats.average_tree_width,
            'pruning_efficiency': self.stats.pruning_efficiency,
        }
