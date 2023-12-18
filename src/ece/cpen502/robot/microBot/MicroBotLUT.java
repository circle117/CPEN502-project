package ece.cpen502.robot.microBot;


import ece.cpen502.LookUpTable;
import ece.cpen502.robot.Action;
import ece.cpen502.robot.State;
import robocode.*;

import java.io.IOException;

public class MicroBotLUT extends MicroBot {

    static private LookUpTable lookUpTable = new LookUpTable(State.EnergyLabel.values().length,
            State.DistanceLabel.values().length,
            State.EnergyLabel.values().length,
            State.DistanceLabel.values().length,
            Action.ActionLabel.values().length);

    private final boolean ifTrain = false;
    private final String fileName = "/data-e0.9-round4000-alpha0.5-gamma0.99-LUT.csv";

    public void run() {
        // robocode setting
        setAdjustRadarForGunTurn(true);             // divorce radar movement from robot movement
        setAdjustGunForRobotTurn(true);             // divorce gun movement from robot movement
        enemy.reset();

        if (ifTrain) {
            if (lookUpTable.getWriter() == null) {
                try {
                    lookUpTable.setWriter(getDataDirectory());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        } else {
            try {
                lookUpTable.load(getDataDirectory().getPath()+fileName);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        // look up table setting
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
        double[] index = state.indexForLUT(this, event);

        if (ifTrain) {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.train(index, new double[]{reward})[0]));
                action.doAction(this, event);
                reward = 0;
            } catch (NullPointerException exception) {
                out.println(exception);
            }
        } else {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.outputFor(index)[0]));
                action.doAction(this, event);
            } catch (NullPointerException exception) {
                out.println(exception);
            }
        }
    }

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        if (ifTrain) {
            if (event.getName().equals(enemy.getName())) {
                reward += goodTerminalReward;
                lookUpTable.win();
                enemy.reset();
            }

            try {
                lookUpTable.terminalState(reward, getTime());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void onDeath(DeathEvent event) {
        if (ifTrain) {
            reward += badTerminalReward;

            try {
                lookUpTable.terminalState(reward, getTime());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
