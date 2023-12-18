package ece.cpen502.robot.microBot;

import ece.cpen502.NeuralNet;
import ece.cpen502.ReplayMemory.Experience;
import ece.cpen502.ReplayMemory.ReplayMemory;
import ece.cpen502.robot.Action;
import robocode.*;

import java.io.IOException;

public class MicroBotNN extends MicroBot {

    // hyper parameter
    private static double epsilon = 0.6;
    private final static double epsilonInitial = 0.6;

    private final int totalRoundNum = 800;
    private final double gamma = 0.99;
    private final int memSize = 200;
    private final int batchSize = 200;
    private final boolean ifTrain = true;

    // NN
    private static NeuralNet net = new NeuralNet(4, 10, 5, 0.1, 0.9,
            0, 1, false);
    private static ReplayMemory<Experience> memory;
    private Experience preExperience;

    // for records
    private final static String NNFileFormat = "/online-weights-round%d-epsilon%.1f-gamma%.2f-memSize%d-batchSize%d.csv";
    private final static String recordFileFormat = "/online-round%d-epsilon%.1f-gamma%.2f-memSize%d-batchSize%d.csv";


    public MicroBotNN() {
        super();
        model = new NeuralNet(4, 10, 5,
                0.1, 0.9, 0, 1, false);
        memory = new ReplayMemory<>(memSize);
    }

    @Override
    public void run() {
        recordFileName = getDataDirectory() +
                String.format(recordFileFormat, totalRoundNum, epsilonInitial, gamma, memSize, batchSize);
        modelFileName = getDataDirectory() +
                String.format(NNFileFormat, totalRoundNum, epsilonInitial, gamma, memSize, batchSize);
        super.run();
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        double turn = getHeading() - getGunHeading() + event.getBearing();
        setTurnGunRight(normalizeBearing(turn));
        if (enemy.none() | event.getName().equals(enemy.getName())) {
            enemy.update(event);
        }

        // store experience
        Experience experience = new Experience();
        int cnt = 0;
        for (double el:state.inputForNN(this, event)) {           // state / nextState
            experience.state[cnt] = el;
            if (preExperience != null)
                preExperience.nextState[cnt] = el;
            cnt++;
        }

        if (ifTrain && preExperience != null) {                         // reward
            preExperience.reward = reward;
            reward = 0;
            memory.add(preExperience);
            rewards.add(reward);
        }

        double[] qValues;
        double random = Math.random();
        if (ifTrain && random < epsilon) {                              // action
            experience.action = Math.floor(Math.random() * Action.ActionLabel.values().length);
        } else {
            qValues = net.outputFor(experience.state);
            experience.action = findMaxAction(qValues);
        }
        action.setLabel(action.labelTransfer(experience.action));
        action.doAction(this, event);
        preExperience = experience;

        if (ifTrain && memory.sizeOf() >= batchSize) {
            Object[] Experiences = memory.randomSample(batchSize);
            for (Object object:Experiences) {
                Experience e = (Experience) object;
                int actionIndex = (int) e.action;
                net.setIndex(actionIndex);
                double[] Y = new double[Action.ActionLabel.values().length];
                if (e.nextState == null) {
                    Y[actionIndex] = e.reward;
                } else {
                    qValues = net.outputFor(e.nextState);
                    int maxAction = (int)findMaxAction(qValues);
                    Y[actionIndex] = e.reward + gamma * qValues[maxAction];
                }
                errors.add(net.train(e.state, Y)[actionIndex]);
            }
        }
    }

    protected void terminal(double terminalReward) throws IOException {
        reward += terminalReward;
        preExperience.reward = reward;
        memory.add(preExperience);
        preExperience = null;
        super.terminal(terminalReward);
    }
}
