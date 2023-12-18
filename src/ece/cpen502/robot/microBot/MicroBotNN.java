package ece.cpen502.robot.microBot;

import ece.cpen502.NeuralNet;
import ece.cpen502.ReplayMemory.Experience;
import ece.cpen502.ReplayMemory.ReplayMemory;
import ece.cpen502.robot.Action;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class MicroBotNN extends MicroBot {

    // hyper parameter
    private final static double epsilonInitial = 0.6;
    private static double epsilon = 0.6;
    private final double epsilonDecay;
    private final int totalRoundNum = 800;
    private final double gamma = 0.99;
    private final int memSize = 200;
    private final int batchSize = 200;
    private final static String NNFileName = "/online-weights-round%d-epsilon%.1f-gamma%.2f-memSize%d-batchSize%d.csv";
    private final boolean ifTrain = true;

    // NN
    private static NeuralNet net = new NeuralNet(4, 10, 5, 0.1, 0.9,
            0, 1, false);
    private static ReplayMemory<Experience> memory;
    private Experience preExperience;

    // for records
    private static int roundNum = 0;
    private final int recordInterval = 25;
    private static int winsPerInterval = 0;
    private static ArrayList<Double> rewards = new ArrayList<>();
    private static ArrayList<Long> timeEveryRound = new ArrayList<>();
    private static ArrayList<Double> errors = new ArrayList<>();
    private final String recordFileName = "/online-round%d-epsilon%.1f-gamma%.2f-memSize%d-batchSize%d.csv";
    private static RobocodeFileWriter writer;


    public MicroBotNN() {
        epsilonDecay = epsilonInitial/(totalRoundNum * 0.8);
        memory = new ReplayMemory<>(memSize);
    }

    public void run() {
        try {
            if (ifTrain && writer == null) {
                writer = new RobocodeFileWriter(getDataDirectory() +
                        String.format(recordFileName, totalRoundNum, epsilon, gamma, memSize, batchSize));
                writer.write(String.join(",",
                        "RoundNumber", "winsPer50", "epsilon", "error",  "meanTotalReward", "meanTime\n"));
            } else if (!ifTrain){
                // load weights
                net.load(getDataDirectory().getPath() + String.format(NNFileName, totalRoundNum,
                        epsilon, gamma, memSize, batchSize));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        setAdjustRadarForGunTurn(true);             // divorce radar movement from robot movement
        setAdjustGunForRobotTurn(true);             // divorce gun movement from robot movement
        enemy.reset();

        while(true) {
            setTurnRadarRight(10000);
            execute();
        }
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

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        if (ifTrain) {
            winsPerInterval++;
            try {
                terminal(goodTerminalReward);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void onDeath(DeathEvent event) {
        if (ifTrain) {
            try {
                terminal(badTerminalReward);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private double findMaxAction(double[] qValues) {
        double maxValue = Double.MIN_VALUE;
        double maxAction = 0;
        for (int i = 0; i < Action.ActionLabel.values().length; i++) {
            if (qValues[i] > maxValue) {
                maxValue = qValues[i];
                maxAction = i;
            }
        }
        return maxAction;
    }

    private void terminal(double terminalReward) throws IOException {
        reward += terminalReward;
        preExperience.reward = reward;
        memory.add(preExperience);

        roundNum++;
        timeEveryRound.add(getTime());
        rewards.add(reward);
        if (ifTrain && roundNum % recordInterval == 0) {
            double meanError = 0;
            for (double element: errors)
                meanError += element;
            meanError /= errors.size();

            double meanReward = 0;
            for (double element: rewards)
                meanReward += element;
            meanReward /= rewards.size();

            long meanTime = 0;
            for (long element: timeEveryRound)
                meanTime += element;
            meanTime /= timeEveryRound.size();

            writer.write(String.join(",",
                    String.valueOf(roundNum), String.valueOf(winsPerInterval*1.0/recordInterval),
                    String.valueOf(epsilon), String.valueOf(meanError),
                    String.valueOf(meanReward), meanTime + "\n"));
            rewards = new ArrayList<>();
            winsPerInterval = 0;
        }

        if (roundNum == totalRoundNum) {
            writer.close();
            net.save(new File(getDataDirectory() + String.format(NNFileName, totalRoundNum,
                    epsilonInitial, gamma, memSize, batchSize)));
        }

        if (epsilon > 0) {
            epsilon -= epsilonDecay;
            if (epsilon <= 0)
                epsilon = 0.0;
        }

        preExperience = null;
        reward = 0;
    }
}
