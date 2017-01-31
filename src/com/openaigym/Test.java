package com.openaigym;



import java.io.IOException;

import org.deeplearning4j.gym.Client;
import org.deeplearning4j.gym.ClientFactory;
import org.deeplearning4j.gym.ClientUtils;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;



public class Test {

	public static void main(String[] args) throws IOException {

		String url = "http://127.0.0.1:5000";
		String env = "CartPole-v0";
		String instanceID = "e15739cf";
		String testDir = "/tmp/testDir";
		boolean render = true;
		String renderStr = render ? "True" : "False";
		
		Client<Box, Integer, DiscreteSpace> client = ClientFactory.build(url, env, render);
		client.monitorStart(testDir, true, false);

		int episodeCount = 1;
		int maxSteps = 200;
		int reward = 0;

		for (int i = 0; i < episodeCount; i++) {
			client.reset();

			for (int j = 0; j < maxSteps; j++) {

				Integer action = client.getActionSpace().randomAction();
				StepReply<Box> step = client.step(action);
				reward += step.getReward();

				if (step.isDone()) {
					// System.out.println("break");
					break;
				}
			}

		}

		client.monitorClose();
		client.upload(testDir, "YOUR_OPENAI_GYM_API_KEY");
	}

}
