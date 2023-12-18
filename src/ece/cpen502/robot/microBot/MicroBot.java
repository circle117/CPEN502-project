package ece.cpen502.robot.microBot;

import ece.cpen502.interfaces.CommonInterface;
import ece.cpen502.robot.Action;
import ece.cpen502.robot.State;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public abstract class MicroBot extends AdvancedRobot {

    protected Action action = new Action();
    protected State state = new State();

    // training parameters
    private final static double epsilonInitial = 0.6;
    private static double epsilon = 0.6;
    private double epsilonDecay;
    protected final int totalRoundNum = 800;
    protected double reward = 0.0;
    protected final double badReward = -0.01;
    protected final double goodReward = 0.01;
    protected final double badTerminalReward = -0.02;
    protected final double goodTerminalReward = 0.02;
    protected final boolean ifTrain = true;
    protected static CommonInterface model;


    // for recording
    private static int roundNum = 0;
    private final int recordInterval = 25;
    private static int winsPerInterval = 0;
    private static ArrayList<Long> timeEveryRound = new ArrayList<>();
    protected static ArrayList<Double> rewards = new ArrayList<>();
    static ArrayList<Double> errors = new ArrayList<>();
    private static RobocodeFileWriter writer;
    protected static String recordFileName;
    protected static String modelFileName;

    // robot setting
    protected final EnemyBot enemy = new EnemyBot();
    protected int fireVariable = 400;
    protected byte moveDirection = 1;

    public MicroBot() {
        epsilonDecay = epsilonInitial/(totalRoundNum * 0.8);
    }

    public void run() {
        try {
            if (ifTrain && writer == null) {
                writer = new RobocodeFileWriter(recordFileName);
                writer.write(String.join(",",
                        "RoundNumber", "winsPer50", "epsilon", "error",  "meanTotalReward", "meanTime\n"));
            } else if (!ifTrain){
                // load weights
                model.load(modelFileName);
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

    protected double findMaxAction(double[] qValues) {
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

    public double normalizeBearing (double angle) {
        while (angle > 180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        if (ifTrain) {
            reward += goodReward;
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        if (ifTrain) {
            reward += badReward;
        }
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        action.setLabel(Action.ActionLabel.HEAD2CENTER);
        action.doAction(this, null);
    }

    public void setMoveDirection(byte moveDirection) {
        this.moveDirection = moveDirection;
    }

    public byte getMoveDirection() {
        return moveDirection;
    }

    public EnemyBot getEnemy() {
        return enemy;
    }

    public int getFireVariable() {
        return fireVariable;
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

    protected void terminal(double terminalReward) throws IOException {
        reward += terminalReward;

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
            timeEveryRound = new ArrayList<>();
            errors = new ArrayList<>();
            winsPerInterval = 0;
        }

        if (roundNum == totalRoundNum) {
            writer.close();
            model.save(new File(getDataDirectory() + modelFileName));
        }

        if (epsilon > 0) {
            epsilon -= epsilonDecay;
            if (epsilon <= 0)
                epsilon = 0.0;
        }

        reward = 0;
    }
}
