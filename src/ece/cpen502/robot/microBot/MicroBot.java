package ece.cpen502.robot.microBot;


import ece.cpen502.LookUpTable;
import ece.cpen502.robot.Action;
import ece.cpen502.robot.State;
import robocode.*;

import java.io.IOException;

public class MicroBot extends AdvancedRobot {

    private final EnemyBot enemy = new EnemyBot();
    private int fireVariable = 400;
    private byte moveDirection = 1;

    static private LookUpTable lookUpTable = new LookUpTable(State.EnergyLabel.values().length,
            State.DistanceLabel.values().length,
            State.EnergyLabel.values().length,
            State.DistanceLabel.values().length,
            Action.ActionLabel.values().length);
    private Action action = new Action();
    private State state = new State();
    private double reward = 0.0;
    private final boolean ifTrain = true;
    private final String fileName = "/LUT.csv";
    public MicroBot(){
    }

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
        double[] index = indexForState(event);

        if (ifTrain) {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.train(index, reward)));
                action.doAction(this, event);
                reward = 0;
            } catch (NullPointerException exception) {
                out.println(exception);
            }
        } else {
            try {
                action.setLabel(action.labelTransfer(lookUpTable.outputFor(index)));
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
                reward += lookUpTable.getGoodTerminal();
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
            reward += lookUpTable.getBadTerminal();

            try {
                lookUpTable.terminalState(reward, getTime());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        if (ifTrain) {
            reward += lookUpTable.getGoodReward();
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        if (ifTrain) {
            reward += lookUpTable.getBadReward();
        }
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        action.setLabel(Action.ActionLabel.HEAD2CENTER);
        action.doAction(this, null);
    }

    public double[] indexForState(ScannedRobotEvent event) {
        double[] index = new double[4];
        index[0] = state.indexForEnergy(getEnergy());
        index[1] = state.indexForDistance(event.getDistance());
        index[2] = state.indexForEnergy(enemy.getEnergy());
        double distance = Math.sqrt(Math.pow(getBattleFieldWidth()/2 - getX(), 2) +
                Math.pow(getBattleFieldHeight()/2 - getY(), 2));
        index[3] = state.indexForDistance(distance);
        return index;
    }

    public double normalizeBearing (double angle) {
        while (angle > 180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
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
}
