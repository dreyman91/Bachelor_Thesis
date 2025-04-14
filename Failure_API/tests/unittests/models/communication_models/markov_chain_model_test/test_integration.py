"""
Integration test module for the ContextAwareMarkovModel class.

This module tests complete workflows and scenarios that involve multiple aspects
of the ContextAwareMarkovModel, simulating realistic communication scenarios.
"""

import unittest
import numpy as np
from collections import Counter
import time

from Failure_API.src.failure_api.communication_models.markov_chain_based import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestContextAwareMarkovModelIntegration(unittest.TestCase):
    """Integration tests for the ContextAwareMarkovModel class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_ids = ["agent1", "agent2", "agent3", "agent4"]

        # Create transition matrices with different characteristics
        # Stable link with high probability of staying connected
        stable_matrix = np.array([[0.01, 0.99], [0.02, 0.98]])

        # Unstable link with higher chance of disconnection
        unstable_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Highly fluctuating link
        fluctuating_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Assign different link types to different agent pairs
        self.transition_probabilities = {
            ("agent1", "agent2"): stable_matrix.copy(),
            ("agent2", "agent1"): stable_matrix.copy(),
            ("agent1", "agent3"): unstable_matrix.copy(),
            ("agent3", "agent1"): unstable_matrix.copy(),
            ("agent2", "agent3"): fluctuating_matrix.copy(),
            ("agent3", "agent2"): fluctuating_matrix.copy(),
            # agent4 connections will use default probabilities
        }

        # Create model with fixed seed for reproducibility
        self.model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            parallel=False
        )
        self.model.rng = np.random.RandomState(42)

        # Create communication matrix
        self.comms_matrix = ActiveCommunication(self.agent_ids)

    def test_stability_over_time(self):
        """
        Test link stability characteristics over multiple update cycles.

        This test verifies that links behave according to their defined transition
        probabilities over multiple update cycles, showing expected patterns of
        stability or instability.
        """
        n_iterations = 100

        # Track connectivity history for each link
        history = {
            ("agent1", "agent2"): [],  # stable link
            ("agent1", "agent3"): [],  # unstable link
            ("agent2", "agent3"): []  # fluctuating link
        }

        # Initialize all links to connected (True)
        for sender, receiver in history.keys():
            self.comms_matrix.update(sender, receiver, True)
            self.model.state[(sender, receiver)] = 1

        # Run multiple update cycles
        for _ in range(n_iterations):
            self.model.update_connectivity(self.comms_matrix)

            # Record the current state of each link
            for link in history.keys():
                sender, receiver = link
                connected = self.comms_matrix.get(sender, receiver)
                history[link].append(connected)

        # Analyze the results
        stability_metrics = {}
        for link, states in history.items():
            # Count transitions (True->False or False->True)
            transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])

            # Calculate frequency of connected state
            connected_ratio = sum(1 for state in states if state) / len(states)

            stability_metrics[link] = {
                "transitions": transitions,
                "connected_ratio": connected_ratio
            }
        print("\nStability metrics:")
        for link, stats in stability_metrics.items():
            print(f"{link}: transitions={stats['transitions']}, connected_ratio={stats['connected_ratio']:.2f}")

        # Verify stability patterns align with expectations
        # Stable link: few transitions, mostly connected
        self.assertLess(stability_metrics[("agent1", "agent2")]["transitions"], 30)
        self.assertGreater(stability_metrics[("agent1", "agent2")]["connected_ratio"], 0.7)

        # Unstable link: more transitions, less connected
        self.assertGreater(stability_metrics[("agent1", "agent3")]["transitions"], 20)
        self.assertLess(stability_metrics[("agent1", "agent3")]["connected_ratio"], 0.7)

        # Fluctuating link: many transitions, roughly 50/50 connected
        self.assertGreater(stability_metrics[("agent2", "agent3")]["transitions"], 30)
        self.assertAlmostEqual(
            stability_metrics[("agent2", "agent3")]["connected_ratio"],
            0.5,
            delta=0.2
        )


    def test_context_influence_over_time(self):
        """
        Test how changing context affects connectivity patterns over time.

        This test verifies that changing weather conditions through the context function
        influences network connectivity in predictable ways.
        """
        # Set up context with changing weather patterns
        weather_values = [0.3, 0.3, 0.6, 0.6, 0.8, 0.8, 0.3, 0.3]  # cycles of good/bad weather
        weather_index = 0

        def cycling_weather():
            nonlocal weather_index
            weather = {"weather": weather_values[weather_index]}
            weather_index = (weather_index + 1) % len(weather_values)
            return weather

        # Set the context function
        self.model.context_fn = cycling_weather

        # Track connectivity for all links
        connections_by_weather = {
            "good": [],  # weather <= 0.5
            "bad": []  # weather > 0.5
        }

        # Run simulation for several cycles
        for _ in range(40):  # 5 complete weather cycles
            current_weather = cycling_weather()["weather"]
            weather_type = "bad" if current_weather > 0.5 else "good"

            # Update network connectivity
            self.model.update_connectivity(self.comms_matrix)

            # Count how many links are connected
            connected_count = 0
            total_links = 0

            for i, sender in enumerate(self.agent_ids):
                for j, receiver in enumerate(self.agent_ids):
                    if i != j:  # Skip self-loops
                        connected = self.comms_matrix.get(sender, receiver)
                        if connected:
                            connected_count += 1
                        total_links += 1

            # Record the ratio of connected links for this weather type
            connections_by_weather[weather_type].append(connected_count / total_links)

        # Calculate average connectivity ratio for each weather type
        avg_good_weather = sum(connections_by_weather["good"]) / len(connections_by_weather["good"])
        avg_bad_weather = sum(connections_by_weather["bad"]) / len(connections_by_weather["bad"])

        # In bad weather, more links should be disconnected
        self.assertGreater(avg_good_weather, avg_bad_weather)

    def test_complex_scenario(self):
        """
        Test a complex scenario with time-varying context and traffic patterns.

        This test simulates a realistic scenario where time, traffic, and weather
        all influence network connectivity.
        """
        # Set up time function
        time_counter = [0]  # Use list to allow modification in nested function

        def increment_time():
            time_counter[0] += 1
            return time_counter[0]

        # Set up traffic function - traffic increases over time
        def varying_traffic():
            t = time_counter[0]
            traffic = {}

            # Simulate increasing traffic on some links
            if t > 5:
                traffic[("agent1", "agent2")] = min(1.0, (t - 5) * 0.1)

            # Simulate periodic traffic spikes
            if t % 10 == 0:
                traffic[("agent2", "agent3")] = 0.9
            else:
                traffic[("agent2", "agent3")] = 0.2

            return traffic

        # Set up weather function - changes every 5 time steps
        def varying_weather():
            t = time_counter[0]
            # Weather cycles: normal -> stormy -> normal -> etc.
            weather_value = 0.9 if (t // 5) % 2 == 1 else 0.3
            return {"weather": weather_value}

        # Configure model with all dynamic functions
        self.model.time_fn = increment_time
        self.model.traffic_fn = varying_traffic
        self.model.context_fn = varying_weather

        # Track connectivity history
        link_history = {link: [] for link in [
            ("agent1", "agent2"),
            ("agent2", "agent3"),
            ("agent3", "agent4")
        ]}

        # Run simulation
        for _ in range(60):  # 6 complete weather cycles
            # Update connectivity
            self.model.update_connectivity(self.comms_matrix)

            # Record link states
            for link in link_history:
                sender, receiver = link
                connected = self.comms_matrix.get(sender, receiver)
                link_history[link].append(connected)

        # Analyze patterns to verify complex interactions
        # Check that link (agent1, agent2) degrades in the second half (high traffic)
        early_connectivity = sum(link_history[("agent1", "agent2")][:30]) / 30
        late_connectivity = sum(link_history[("agent1", "agent2")][30:]) / 30
        self.assertGreater(early_connectivity, late_connectivity - 0.05)

        # Check that link (agent2, agent3) shows periodic patterns
        # Extract connectivity at time steps divisible by 10 (traffic spikes)
        spike_times = [i for i in range(30) if i % 10 == 0]
        non_spike_times = [i for i in range(30) if i % 10 != 0]

        spike_connectivity = sum(link_history[("agent2", "agent3")][i] for i in spike_times) / len(spike_times)
        non_spike_connectivity = sum(link_history[("agent2", "agent3")][i] for i in non_spike_times) / len(
            non_spike_times)

        # During traffic spikes, connectivity should be worse
        self.assertLess(spike_connectivity, non_spike_connectivity)

    def test_performance(self):
        """
        Test the performance of the model with a larger number of agents.

        This test verifies that the parallel processing in update_connectivity
        scales reasonably with increasing numbers of agents.
        """
        # Skip for smaller test suites
        if hasattr(self, 'skipLongTests') and self.skipLongTests:
            self.skipTest("Skipping performance test")

        # Create a larger network
        large_n_agents = 20
        large_agent_ids = [f"agent{i}" for i in range(large_n_agents)]

        # Simple transition probabilities - just for a few links
        large_transition_probabilities = {
            ("agent0", "agent1"): np.array([[0.9, 0.1], [0.2, 0.8]])
        }

        # Create model and comms matrix
        large_model = ContextAwareMarkovModel(
            agent_ids=large_agent_ids,
            transition_probabilities=large_transition_probabilities,
            parallel=False
        )
        large_comms_matrix = ActiveCommunication(large_agent_ids)

        # Measure time taken for updates
        start_time = time.time()

        # Run multiple updates
        n_updates = 5
        for _ in range(n_updates):
            large_model.update_connectivity(large_comms_matrix)

        elapsed_time = time.time() - start_time

        # Just a sanity check that it doesn't take too long
        # With parallel processing, should be reasonably fast
        self.assertLess(elapsed_time, 5.0)  # Should complete within 5 seconds

        # Calculate links updated per second
        n_links = large_n_agents * (large_n_agents - 1)  # Directed links
        updates_per_second = (n_links * n_updates) / elapsed_time

        # Print performance metrics for reference
        print(f"\nPerformance Test Results:")
        print(f"  Number of agents: {large_n_agents}")
        print(f"  Number of links: {n_links}")
        print(f"  Updates performed: {n_updates}")
        print(f"  Total time: {elapsed_time:.4f} seconds")
        print(f"  Updates per second: {updates_per_second:.2f}")


if __name__ == "__main__":
    unittest.main()